"""
Perceiver 모델 학습 스크립트
다양한 설정과 함께 모델을 학습시키는 메인 스크립트
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

from perceiver_model import PerceiverModel, create_perceiver_model
from data_utils import (
    create_synthetic_sequence_dataloader, 
    create_cifar10_sequence_dataset
)


class TrainingConfig:
    """학습 설정 클래스"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        # 기본 설정
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.num_epochs = 100
        self.weight_decay = 1e-4
        self.warmup_steps = 1000
        self.gradient_clip_norm = 1.0
        
        # 모델 설정
        self.model_size = 'base'  # 'small', 'base', 'large'
        self.input_dim = 32
        self.num_classes = 10
        
        # 데이터 설정
        self.dataset_type = 'synthetic'  # 'synthetic', 'cifar10'
        self.seq_len_range = (50, 200)
        self.num_samples = 5000
        self.patch_size = 8
        
        # 학습 설정
        self.save_dir = './checkpoints'
        self.log_dir = './logs'
        self.save_every = 10
        self.eval_every = 5
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 설정 덮어쓰기
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        # 디렉토리 생성
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


class PerceiverTrainer:
    """Perceiver 모델 학습 클래스"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 로깅 설정
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(
            os.path.join(config.log_dir, f'perceiver_{timestamp}')
        )
        
        # 모델 초기화
        self.model = self._create_model()
        self.model.to(self.device)
        
        # 옵티마이저 설정
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 스케줄러 설정 (Cosine Annealing with Warmup)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.1
        )
        
        # 손실 함수
        self.criterion = nn.CrossEntropyLoss()
        
        # 학습 상태 추적
        self.global_step = 0
        self.best_accuracy = 0.0
        self.train_losses = []
        self.val_accuracies = []
        
        print(f"모델 파라미터 수: {self.count_parameters():,}")
        print(f"사용 장치: {self.device}")
        
    def _create_model(self) -> PerceiverModel:
        """모델 생성"""
        return create_perceiver_model(
            input_dim=self.config.input_dim,
            num_classes=self.config.num_classes,
            model_size=self.config.model_size
        )
    
    def count_parameters(self) -> int:
        """학습 가능한 파라미터 수 계산"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _warmup_lr(self, step: int) -> float:
        """Warmup 스케줄러"""
        if step < self.config.warmup_steps:
            return float(step) / float(max(1, self.config.warmup_steps))
        return 1.0
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # 데이터를 GPU로 이동
            inputs = batch['input'].to(self.device)
            labels = batch['label'].to(self.device)
            masks = batch['mask'].to(self.device) if 'mask' in batch else None
            
            # Warmup
            if self.global_step < self.config.warmup_steps:
                lr_scale = self._warmup_lr(self.global_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.config.learning_rate * lr_scale
            
            # 순전파
            self.optimizer.zero_grad()
            logits = self.model(inputs, masks)
            loss = self.criterion(logits, labels)
            
            # 역전파
            loss.backward()
            
            # 그래디언트 클리핑
            if self.config.gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip_norm
                )
            
            self.optimizer.step()
            
            # 통계 업데이트
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # 로깅
            if batch_idx % 50 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'에포크 [{epoch+1}/{self.config.num_epochs}] '
                      f'배치 [{batch_idx}/{len(dataloader)}] '
                      f'손실: {loss.item():.4f} '
                      f'학습률: {current_lr:.2e}')
                
                self.writer.add_scalar('Train/BatchLoss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/LearningRate', current_lr, self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / total_samples
        return avg_loss
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """모델 평가"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['input'].to(self.device)
                labels = batch['label'].to(self.device)
                masks = batch['mask'].to(self.device) if 'mask' in batch else None
                
                # 순전파
                logits = self.model(inputs, masks)
                loss = self.criterion(logits, labels)
                
                # 예측
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        accuracy = correct / total_samples
        avg_loss = total_loss / total_samples
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'config': vars(self.config)
        }
        
        # 일반 체크포인트 저장
        checkpoint_path = os.path.join(
            self.config.save_dir, 
            f'checkpoint_epoch_{epoch+1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 모델 저장
        if is_best:
            best_path = os.path.join(self.config.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"새로운 최고 모델 저장! 정확도: {self.best_accuracy:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_accuracy = checkpoint['best_accuracy']
        
        print(f"체크포인트 로드 완료: {checkpoint_path}")
        return checkpoint['epoch']
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """전체 학습 프로세스"""
        print("학습 시작!")
        print(f"학습 데이터: {len(train_loader)} 배치")
        print(f"검증 데이터: {len(val_loader)} 배치")
        
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            
            # 학습
            train_loss = self.train_epoch(train_loader, epoch)
            self.train_losses.append(train_loss)
            
            # 스케줄러 스텝
            if self.global_step >= self.config.warmup_steps:
                self.scheduler.step()
            
            # 검증
            if (epoch + 1) % self.config.eval_every == 0:
                val_metrics = self.evaluate(val_loader)
                val_accuracy = val_metrics['accuracy']
                val_loss = val_metrics['loss']
                
                self.val_accuracies.append(val_accuracy)
                
                # 로깅
                epoch_time = time.time() - start_time
                print(f'\n에포크 [{epoch+1}/{self.config.num_epochs}] 완료')
                print(f'학습 손실: {train_loss:.4f}')
                print(f'검증 손실: {val_loss:.4f}')
                print(f'검증 정확도: {val_accuracy:.4f}')
                print(f'소요 시간: {epoch_time:.2f}초\n')
                
                # TensorBoard 로깅
                self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Val/Accuracy', val_accuracy, epoch)
                
                # 최고 모델 체크
                is_best = val_accuracy > self.best_accuracy
                if is_best:
                    self.best_accuracy = val_accuracy
                
                # 체크포인트 저장
                if (epoch + 1) % self.config.save_every == 0:
                    self.save_checkpoint(epoch, is_best)
        
        print(f"학습 완료! 최고 정확도: {self.best_accuracy:.4f}")
        self.writer.close()


def create_data_loaders(config: TrainingConfig):
    """데이터 로더 생성"""
    if config.dataset_type == 'synthetic':
        print("합성 시퀀스 데이터셋 사용")
        train_loader = create_synthetic_sequence_dataloader(
            batch_size=config.batch_size,
            num_samples=config.num_samples,
            seq_len_range=config.seq_len_range,
            feature_dim=config.input_dim,
            num_classes=config.num_classes
        )
        
        val_loader = create_synthetic_sequence_dataloader(
            batch_size=config.batch_size,
            num_samples=config.num_samples // 5,  # 검증용은 더 적게
            seq_len_range=config.seq_len_range,
            feature_dim=config.input_dim,
            num_classes=config.num_classes
        )
        
    elif config.dataset_type == 'cifar10':
        print("CIFAR-10 패치 시퀀스 데이터셋 사용")
        train_loader = create_cifar10_sequence_dataset(
            batch_size=config.batch_size,
            patch_size=config.patch_size,
            train=True
        )
        
        val_loader = create_cifar10_sequence_dataset(
            batch_size=config.batch_size,
            patch_size=config.patch_size,
            train=False
        )
        
        # CIFAR-10의 경우 입력 차원 업데이트
        config.input_dim = config.patch_size * config.patch_size * 3
        config.num_classes = 10
        
    else:
        raise ValueError(f"지원하지 않는 데이터셋 타입: {config.dataset_type}")
    
    return train_loader, val_loader


def main():
    """메인 학습 함수"""
    parser = argparse.ArgumentParser(description='Perceiver 모델 학습')
    parser.add_argument('--config', type=str, default=None,
                       help='JSON 설정 파일 경로')
    parser.add_argument('--dataset', type=str, default='synthetic',
                       choices=['synthetic', 'cifar10'], help='데이터셋 종류')
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['small', 'base', 'large'], help='모델 크기')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=50, help='에포크 수')
    parser.add_argument('--lr', type=float, default=1e-4, help='학습률')
    parser.add_argument('--resume', type=str, default=None,
                       help='재개할 체크포인트 경로')
    
    args = parser.parse_args()
    
    # 설정 로드
    config_dict = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    
    # 명령줄 인수로 설정 덮어쓰기
    config_dict.update({
        'dataset_type': args.dataset,
        'model_size': args.model_size,
        'batch_size': args.batch_size,
        'num_epochs': args.epochs,
        'learning_rate': args.lr
    })
    
    config = TrainingConfig(config_dict)
    
    # 설정 출력
    print("=== 학습 설정 ===")
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 30)
    
    # 데이터 로더 생성
    train_loader, val_loader = create_data_loaders(config)
    
    # 트레이너 생성
    trainer = PerceiverTrainer(config)
    
    # 체크포인트에서 재개
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # 학습 시작
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()