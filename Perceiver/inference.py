"""
Perceiver 모델 추론 및 평가 스크립트
학습된 모델을 사용해 추론하고 성능을 평가하는 스크립트
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import json
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from perceiver_model import create_perceiver_model
from data_utils import (
    create_synthetic_sequence_dataloader,
    create_cifar10_sequence_dataset
)


class PerceiverInference:
    """Perceiver 모델 추론 클래스"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """
        Args:
            model_path: 학습된 모델 체크포인트 경로
            device: 사용할 디바이스 ('auto', 'cpu', 'cuda')
        """
        # 디바이스 설정
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"추론 디바이스: {self.device}")
        
        # 체크포인트 로드
        self.checkpoint = torch.load(model_path, map_location=self.device)
        self.config = self.checkpoint['config']
        
        # 모델 생성 및 로드
        self.model = create_perceiver_model(
            input_dim=self.config['input_dim'],
            num_classes=self.config['num_classes'],
            model_size=self.config['model_size']
        )
        
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        print(f"모델 로드 완료: {model_path}")
        print(f"모델 설정: {self.config}")
        print(f"최고 정확도: {self.checkpoint.get('best_accuracy', 'N/A')}")
    
    def predict_batch(self, 
                     inputs: torch.Tensor, 
                     masks: Optional[torch.Tensor] = None,
                     return_probabilities: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        배치 데이터에 대한 예측
        
        Args:
            inputs: 입력 텐서 (batch_size, seq_len, input_dim)
            masks: 마스크 텐서
            return_probabilities: 확률값 반환 여부
            
        Returns:
            predictions: 예측 클래스
            probabilities: 클래스별 확률 (return_probabilities=True인 경우)
        """
        inputs = inputs.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)
        
        with torch.no_grad():
            logits = self.model(inputs, masks)
            predictions = torch.argmax(logits, dim=-1)
            
            probabilities = None
            if return_probabilities:
                probabilities = F.softmax(logits, dim=-1)
        
        return predictions.cpu(), probabilities.cpu() if probabilities is not None else None
    
    def predict_single(self, 
                      input_data: torch.Tensor,
                      mask: Optional[torch.Tensor] = None,
                      return_probabilities: bool = True) -> Dict:
        """
        단일 샘플에 대한 예측
        
        Args:
            input_data: 입력 데이터 (seq_len, input_dim)
            mask: 마스크 (seq_len,)
            return_probabilities: 확률값 반환 여부
            
        Returns:
            예측 결과 딕셔너리
        """
        # 배치 차원 추가
        input_batch = input_data.unsqueeze(0)
        mask_batch = mask.unsqueeze(0) if mask is not None else None
        
        # 예측
        predictions, probabilities = self.predict_batch(
            input_batch, mask_batch, return_probabilities
        )
        
        result = {
            'prediction': predictions.item(),
            'input_shape': input_data.shape
        }
        
        if return_probabilities:
            result['probabilities'] = probabilities.squeeze(0).numpy()
            result['confidence'] = probabilities.max().item()
        
        return result
    
    def evaluate_dataset(self, dataloader: DataLoader) -> Dict:
        """
        데이터셋 전체에 대한 평가
        
        Args:
            dataloader: 평가할 데이터로더
            
        Returns:
            평가 결과 딕셔너리
        """
        all_predictions = []
        all_labels = []
        all_probabilities = []
        total_inference_time = 0
        
        print("데이터셋 평가 중...")
        for batch_idx, batch in enumerate(dataloader):
            inputs = batch['input']
            labels = batch['label']
            masks = batch['mask'] if 'mask' in batch else None
            
            # 추론 시간 측정
            start_time = time.time()
            predictions, probabilities = self.predict_batch(
                inputs, masks, return_probabilities=True
            )
            inference_time = time.time() - start_time
            total_inference_time += inference_time
            
            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.append(probabilities.numpy())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"처리된 배치: {batch_idx + 1}/{len(dataloader)}")
        
        # 결과 계산
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.vstack(all_probabilities)
        
        accuracy = (all_predictions == all_labels).mean()
        
        # 클래스별 정확도
        num_classes = self.config['num_classes']
        class_accuracies = {}
        for cls in range(num_classes):
            mask = all_labels == cls
            if mask.sum() > 0:
                class_accuracies[cls] = (all_predictions[mask] == all_labels[mask]).mean()
        
        # 신뢰도 분석
        confidences = all_probabilities.max(axis=1)
        
        results = {
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'confidences': confidences,
            'avg_inference_time_per_batch': total_inference_time / len(dataloader),
            'total_samples': len(all_labels)
        }
        
        return results
    
    def generate_confusion_matrix(self, predictions: np.ndarray, labels: np.ndarray, 
                                save_path: Optional[str] = None) -> np.ndarray:
        """
        Confusion Matrix 생성 및 시각화
        
        Args:
            predictions: 예측값
            labels: 실제 라벨
            save_path: 저장 경로 (None이면 화면에 표시)
            
        Returns:
            confusion matrix
        """
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion Matrix 저장: {save_path}")
        else:
            plt.show()
        
        return cm
    
    def analyze_predictions(self, results: Dict, save_dir: Optional[str] = None):
        """
        예측 결과 상세 분석
        
        Args:
            results: evaluate_dataset의 결과
            save_dir: 분석 결과 저장 디렉토리
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        print("\n=== 예측 결과 분석 ===")
        print(f"전체 정확도: {results['accuracy']:.4f}")
        print(f"총 샘플 수: {results['total_samples']}")
        print(f"평균 추론 시간 (배치당): {results['avg_inference_time_per_batch']:.4f}초")
        
        # 클래스별 정확도
        print("\n클래스별 정확도:")
        for cls, acc in results['class_accuracies'].items():
            print(f"클래스 {cls}: {acc:.4f}")
        
        # 신뢰도 분석
        confidences = results['confidences']
        print(f"\n신뢰도 통계:")
        print(f"평균 신뢰도: {confidences.mean():.4f}")
        print(f"신뢰도 표준편차: {confidences.std():.4f}")
        print(f"최소 신뢰도: {confidences.min():.4f}")
        print(f"최대 신뢰도: {confidences.max():.4f}")
        
        # Classification Report
        print(f"\n분류 보고서:")
        print(classification_report(results['labels'], results['predictions']))
        
        # Confusion Matrix
        cm_path = os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
        self.generate_confusion_matrix(
            results['predictions'], 
            results['labels'], 
            cm_path
        )
        
        # 신뢰도 히스토그램
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=30, alpha=0.7, edgecolor='black')
        plt.title('예측 신뢰도 분포')
        plt.xlabel('신뢰도')
        plt.ylabel('빈도')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'confidence_histogram.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"신뢰도 히스토그램 저장: {save_dir}/confidence_histogram.png")
        else:
            plt.show()
        
        # 정확도 vs 신뢰도 분석
        correct_predictions = (results['predictions'] == results['labels'])
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        correct_confidences = confidences[correct_predictions]
        wrong_confidences = confidences[~correct_predictions]
        
        plt.hist(correct_confidences, bins=20, alpha=0.7, label='정답', color='green')
        plt.hist(wrong_confidences, bins=20, alpha=0.7, label='오답', color='red')
        plt.xlabel('신뢰도')
        plt.ylabel('빈도')
        plt.title('정답/오답별 신뢰도 분포')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        # 신뢰도 구간별 정확도
        confidence_bins = np.linspace(0, 1, 11)
        accuracies_by_confidence = []
        
        for i in range(len(confidence_bins) - 1):
            mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
            if mask.sum() > 0:
                acc = correct_predictions[mask].mean()
                accuracies_by_confidence.append(acc)
            else:
                accuracies_by_confidence.append(0)
        
        bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
        plt.plot(bin_centers, accuracies_by_confidence, 'o-')
        plt.xlabel('신뢰도 구간')
        plt.ylabel('정확도')
        plt.title('신뢰도별 정확도')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'confidence_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            print(f"신뢰도 분석 저장: {save_dir}/confidence_analysis.png")
        else:
            plt.show()


def create_test_dataloader(config: Dict, batch_size: int = 32) -> DataLoader:
    """테스트 데이터로더 생성"""
    if config['dataset_type'] == 'synthetic':
        return create_synthetic_sequence_dataloader(
            batch_size=batch_size,
            num_samples=500,  # 테스트용 샘플 수
            seq_len_range=config['seq_len_range'],
            feature_dim=config['input_dim'],
            num_classes=config['num_classes']
        )
    elif config['dataset_type'] == 'cifar10':
        return create_cifar10_sequence_dataset(
            batch_size=batch_size,
            patch_size=config['patch_size'],
            train=False  # 테스트 데이터 사용
        )
    else:
        raise ValueError(f"지원하지 않는 데이터셋: {config['dataset_type']}")


def main():
    """메인 추론 함수"""
    parser = argparse.ArgumentParser(description='Perceiver 모델 추론')
    parser.add_argument('--model_path', type=str, required=True,
                       help='학습된 모델 체크포인트 경로')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='배치 크기')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'], help='사용할 디바이스')
    parser.add_argument('--output_dir', type=str, default='./inference_results',
                       help='결과 저장 디렉토리')
    parser.add_argument('--analyze', action='store_true',
                       help='상세 분석 수행')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"모델 파일을 찾을 수 없습니다: {args.model_path}")
        return
    
    # 추론기 생성
    inferencer = PerceiverInference(args.model_path, args.device)
    
    # 테스트 데이터로더 생성
    test_loader = create_test_dataloader(inferencer.config, args.batch_size)
    
    # 평가 수행
    print("모델 평가 시작...")
    results = inferencer.evaluate_dataset(test_loader)
    
    # 기본 결과 출력
    print(f"\n=== 평가 결과 ===")
    print(f"정확도: {results['accuracy']:.4f}")
    print(f"평균 신뢰도: {results['confidences'].mean():.4f}")
    
    # 상세 분석
    if args.analyze:
        os.makedirs(args.output_dir, exist_ok=True)
        inferencer.analyze_predictions(results, args.output_dir)
        
        # 결과를 JSON으로 저장
        results_to_save = {
            'accuracy': float(results['accuracy']),
            'class_accuracies': {str(k): float(v) for k, v in results['class_accuracies'].items()},
            'avg_confidence': float(results['confidences'].mean()),
            'confidence_std': float(results['confidences'].std()),
            'total_samples': int(results['total_samples']),
            'avg_inference_time_per_batch': float(results['avg_inference_time_per_batch'])
        }
        
        with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"상세 분석 결과가 저장되었습니다: {args.output_dir}")


if __name__ == "__main__":
    main()