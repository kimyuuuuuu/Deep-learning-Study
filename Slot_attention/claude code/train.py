import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import argparse
from datetime import datetime

from slot_attention import SlotAttentionAutoEncoder
from dataset import create_dataloader
from visualize import visualize_slots, save_reconstruction_examples


class SlotAttentionTrainer:
    """
    Slot Attention 모델 훈련 클래스
    """
    
    def __init__(self, config):
        """
        Args:
            config: 훈련 설정을 담은 딕셔너리
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"사용 디바이스: {self.device}")
        
        # 모델 초기화
        self.model = SlotAttentionAutoEncoder(
            resolution=config['resolution'],
            num_slots=config['num_slots'],
            num_iterations=config['num_iterations'],
            slot_dim=config['slot_dim'],
            hidden_dim=config['hidden_dim']
        ).to(self.device)
        
        print(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 손실 함수 (MSE)
        self.criterion = nn.MSELoss()
        
        # 옵티마이저
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0)
        )
        
        # 학습률 스케줄러
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=config.get('lr_decay_step', 100),
            gamma=config.get('lr_decay_gamma', 0.5)
        )
        
        # 데이터로더 생성
        self.train_loader, self.val_loader = create_dataloader(
            dataset_type=config['dataset_type'],
            batch_size=config['batch_size'],
            resolution=config['resolution'],
            num_workers=config.get('num_workers', 4),
            train_samples=config.get('train_samples', 10000),
            val_samples=config.get('val_samples', 1000)
        )
        
        # 로깅 설정
        if config.get('log_dir') is None:
            self.log_dir = f'./logs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        else:
            self.log_dir = config['log_dir']
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        
        # 체크포인트 디렉토리
        self.checkpoint_dir = os.path.join(self.log_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 훈련 통계
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self, epoch):
        """
        한 에폭 훈련
        """
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, images in enumerate(progress_bar):
            images = images.to(self.device)
            batch_size = images.size(0)
            
            # Forward pass
            recons, masks, slots, attn = self.model(images)
            
            # 재구성 손실 계산
            loss = self.criterion(recons, images)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # 그래디언트 클리핑 (옵션)
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 진행상황 업데이트
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # TensorBoard 로깅 (배치별)
            global_step = epoch * num_batches + batch_idx
            if global_step % self.config.get('log_interval', 100) == 0:
                self.writer.add_scalar('Loss/Train_Step', loss.item(), global_step)
                
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        # 학습률 스케줄러 업데이트
        self.scheduler.step()
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """
        검증 단계
        """
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        # 시각화를 위한 샘플 저장
        sample_images = None
        sample_recons = None
        sample_masks = None
        
        with torch.no_grad():
            for batch_idx, images in enumerate(self.val_loader):
                images = images.to(self.device)
                
                # Forward pass
                recons, masks, slots, attn = self.model(images)
                
                # 손실 계산
                loss = self.criterion(recons, images)
                total_loss += loss.item()
                
                # 첫 번째 배치의 결과를 시각화용으로 저장
                if batch_idx == 0:
                    sample_images = images[:4].cpu()  # 처음 4개 샘플
                    sample_recons = recons[:4].cpu()
                    sample_masks = masks[:4].cpu()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        # TensorBoard에 검증 결과 로깅
        self.writer.add_scalar('Loss/Validation', avg_loss, epoch)
        
        # 시각화 결과 저장
        if sample_images is not None:
            # 재구성 결과 시각화
            fig = visualize_slots(
                sample_images, 
                sample_recons, 
                sample_masks,
                save_path=os.path.join(self.log_dir, f'reconstruction_epoch_{epoch+1}.png')
            )
            plt.close(fig)
            
            # TensorBoard에 이미지 로깅
            self.writer.add_images('Original', sample_images, epoch)
            self.writer.add_images('Reconstructed', sample_recons, epoch)
            
            # 마스크들을 시각화 (각 슬롯별로)
            for slot_idx in range(sample_masks.size(1)):
                mask = sample_masks[:, slot_idx:slot_idx+1]  # (B, 1, H, W)
                mask_rgb = mask.expand(-1, 3, -1, -1)  # (B, 3, H, W)
                self.writer.add_images(f'Slot_{slot_idx}_Mask', mask_rgb, epoch)
        
        return avg_loss
    
    def train(self, num_epochs):
        """
        전체 훈련 루프
        """
        print(f"훈련 시작 - 총 {num_epochs} 에폭")
        print(f"배치 크기: {self.config['batch_size']}")
        print(f"훈련 샘플: {len(self.train_loader.dataset):,}")
        print(f"검증 샘플: {len(self.val_loader.dataset):,}")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            # 훈련
            train_loss = self.train_epoch(epoch)
            
            # 검증
            val_loss = self.validate_epoch(epoch)
            
            # 로깅
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss:   {val_loss:.4f}')
            print(f'  LR:         {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # TensorBoard 로깅
            self.writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            # 최고 성능 모델 저장
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, is_best=True)
                print(f'  새로운 최고 성능! (Val Loss: {val_loss:.4f})')
            
            # 정기적으로 체크포인트 저장
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch)
            
            print("-" * 50)
        
        # 훈련 완료 후 손실 곡선 저장
        self.plot_training_curves()
        
        print("훈련 완료!")
        print(f"로그 디렉토리: {self.log_dir}")
        print(f"최고 검증 손실: {self.best_val_loss:.4f}")
        
    def save_checkpoint(self, epoch, is_best=False):
        """
        모델 체크포인트 저장
        """
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # 일반 체크포인트
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_epoch_{epoch+1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # 최고 성능 모델
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        """
        체크포인트 로드
        """
        print(f"체크포인트 로드 중: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"체크포인트 로드 완료 - Epoch: {checkpoint['epoch']}")
        return checkpoint['epoch']
    
    def plot_training_curves(self):
        """
        훈련 곡선 시각화
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        ax.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        ax.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)
        
        # 최고 성능 지점 표시
        best_epoch = np.argmin(self.val_losses) + 1
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
        ax.text(best_epoch, self.best_val_loss, 
                f'Best: Epoch {best_epoch}\nLoss: {self.best_val_loss:.4f}',
                verticalalignment='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, 'training_curves.png'))
        plt.close()


def get_default_config():
    """
    기본 훈련 설정
    """
    return {
        # 모델 설정
        'resolution': 64,           # 이미지 해상도
        'num_slots': 7,            # 슬롯 개수
        'num_iterations': 3,       # 슬롯 어텐션 반복 횟수
        'slot_dim': 64,           # 슬롯 차원
        'hidden_dim': 64,         # 숨겨진 레이어 차원
        
        # 데이터 설정
        'dataset_type': 'shapes',  # 'shapes' 또는 'clevr6'
        'batch_size': 32,         # 배치 크기
        'num_workers': 4,         # 데이터 로더 워커 수
        'train_samples': 10000,   # 훈련 샘플 수 (shapes 데이터셋용)
        'val_samples': 1000,      # 검증 샘플 수 (shapes 데이터셋용)
        
        # 훈련 설정
        'learning_rate': 4e-4,    # 학습률
        'weight_decay': 0,        # L2 정규화
        'grad_clip': 0.05,        # 그래디언트 클리핑
        'lr_decay_step': 100,     # 학습률 감소 주기
        'lr_decay_gamma': 0.5,    # 학습률 감소율
        
        # 로깅 설정
        'log_interval': 100,      # 로그 출력 간격 (스텝 단위)
        'save_interval': 10,      # 체크포인트 저장 간격 (에폭 단위)
        'log_dir': None,          # 로그 디렉토리 (None이면 자동 생성)
    }


def main():
    """
    메인 훈련 함수
    """
    parser = argparse.ArgumentParser(description='Slot Attention 훈련')
    parser.add_argument('--epochs', type=int, default=100, help='훈련 에폭 수')
    parser.add_argument('--batch_size', type=int, default=32, help='배치 크기')
    parser.add_argument('--lr', type=float, default=4e-4, help='학습률')
    parser.add_argument('--num_slots', type=int, default=7, help='슬롯 개수')
    parser.add_argument('--resolution', type=int, default=64, help='이미지 해상도')
    parser.add_argument('--dataset', type=str, default='shapes', choices=['shapes', 'clevr6'], help='데이터셋 타입')
    parser.add_argument('--resume', type=str, default=None, help='체크포인트 경로 (훈련 재개시)')
    
    args = parser.parse_args()
    
    # 설정 준비
    config = get_default_config()
    config.update({
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'num_slots': args.num_slots,
        'resolution': args.resolution,
        'dataset_type': args.dataset
    })
    
    # 트레이너 생성
    trainer = SlotAttentionTrainer(config)
    
    # 훈련 재개
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # 훈련 실행
    remaining_epochs = args.epochs - start_epoch
    if remaining_epochs > 0:
        trainer.train(remaining_epochs)
    else:
        print("이미 요청된 에폭 수만큼 훈련이 완료되었습니다.")


if __name__ == '__main__':
    main()