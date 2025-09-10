"""
Perceiver 모델 내부 동작 시각화 도구
입력 데이터가 모델을 통과하면서 어떻게 변화하는지 시각화
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import os

from perceiver_model import PerceiverModel, create_perceiver_model
from data_utils import create_synthetic_sequence_dataloader


class ModelVisualizer:
    """모델 내부 동작을 시각화하는 클래스"""
    
    def __init__(self, model: PerceiverModel, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.activations = {}
        self.attention_weights = {}
        self.hooks = []
        
        # 시각화를 위해 평가 모드로 설정
        self.model.eval()
        
    def register_hooks(self):
        """모델의 각 레이어에 hook을 등록하여 중간 결과 캡처"""
        
        def save_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    self.activations[name] = output[0].detach().cpu()
                else:
                    self.activations[name] = output.detach().cpu()
            return hook
        
        def save_attention_weights(name):
            def hook(module, input, output):
                # MultiHeadAttention의 경우 attention weights 저장
                if hasattr(module, 'last_attention_weights'):
                    self.attention_weights[name] = module.last_attention_weights.detach().cpu()
            return hook
        
        # 입력 투영 후 활성화
        hook = self.model.input_projection.register_forward_hook(
            save_activation('input_projection')
        )
        self.hooks.append(hook)
        
        # Latent embeddings
        self.activations['latent_init'] = self.model.encoder.latent_embeddings.detach().cpu()
        
        # Cross-attention 레이어들
        for i, layer in enumerate(self.model.encoder.cross_attention_layers):
            hook = layer.register_forward_hook(
                save_activation(f'cross_attention_{i}')
            )
            self.hooks.append(hook)
            
            # Cross-attention weights 캡처
            att_hook = layer.cross_attention.register_forward_hook(
                save_attention_weights(f'cross_attention_weights_{i}')
            )
            self.hooks.append(att_hook)
            
        # Self-attention 레이어들  
        for i, layer in enumerate(self.model.encoder.self_attention_layers):
            hook = layer.register_forward_hook(
                save_activation(f'self_attention_{i}')
            )
            self.hooks.append(hook)
            
            # Self-attention weights 캡처
            att_hook = layer.self_attention.register_forward_hook(
                save_attention_weights(f'self_attention_weights_{i}')
            )
            self.hooks.append(att_hook)
        
        # 최종 분류기 전 pooled representation
        hook = self.model.pool.register_forward_hook(
            save_activation('pooled')
        )
        self.hooks.append(hook)
    
    def remove_hooks(self):
        """등록된 hook들을 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def forward_with_visualization(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """시각화를 위한 forward pass"""
        self.activations = {}
        self.attention_weights = {}
        
        with torch.no_grad():
            x = x.to(self.device)
            if mask is not None:
                mask = mask.to(self.device)
            
            output = self.model(x, mask)
            
        return output, self.activations, self.attention_weights
    
    def plot_latent_evolution(self, activations: Dict, save_path: str = None):
        """Latent space의 변화 과정 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Latent Space Evolution Through Perceiver Layers', fontsize=16)
        
        # 각 단계별 latent 표현
        stages = [
            ('latent_init', 'Initial Latent Embeddings'),
            ('cross_attention_0', 'After Cross-Attention'),
            ('self_attention_0', 'After 1st Self-Attention'),
            ('self_attention_2', 'After 3rd Self-Attention'), 
            ('self_attention_5', 'After Final Self-Attention'),
            ('pooled', 'After Global Pooling')
        ]
        
        for idx, (key, title) in enumerate(stages):
            row, col = idx // 3, idx % 3
            
            if key in activations:
                data = activations[key]
                if len(data.shape) == 3:  # (batch, seq, dim)
                    # 첫 번째 배치의 데이터 사용
                    data = data[0].numpy()
                elif len(data.shape) == 2:  # (seq, dim) 또는 (batch, dim)
                    if key == 'latent_init':
                        data = data.numpy()
                    else:
                        data = data[0:1].numpy() if data.shape[0] > 1 else data.numpy()
                
                # 히트맵으로 시각화
                if len(data.shape) == 2:
                    im = axes[row, col].imshow(data.T, cmap='RdBu_r', aspect='auto')
                    axes[row, col].set_xlabel('Sequence Position' if data.shape[0] > 1 else 'Latent Dimension')
                    axes[row, col].set_ylabel('Feature Dimension')
                else:
                    # 1D 데이터인 경우
                    axes[row, col].plot(data)
                    axes[row, col].set_xlabel('Feature Dimension')
                    axes[row, col].set_ylabel('Activation Value')
                
                axes[row, col].set_title(title)
                plt.colorbar(im, ax=axes[row, col])
            else:
                axes[row, col].set_title(f'{title} (Not Available)')
                axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Latent evolution plot saved to {save_path}")
        plt.show()
    
    def plot_attention_patterns(self, activations: Dict, save_path: str = None):
        """Attention 패턴 시각화"""
        # Cross-attention과 Self-attention 분석
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Input Sequence Norms', 'Latent Evolution', 
                           'Cross-Attention Impact', 'Self-Attention Evolution'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. 입력 시퀀스의 노름 변화
        if 'input_projection' in activations:
            input_norms = torch.norm(activations['input_projection'][0], dim=-1).numpy()
            fig.add_trace(
                go.Scatter(y=input_norms, mode='lines+markers', name='Input Norms'),
                row=1, col=1
            )
        
        # 2. Latent의 노름 변화 추적
        latent_keys = [k for k in activations.keys() if 'attention' in k or 'latent' in k]
        for key in sorted(latent_keys):
            if activations[key].dim() == 3:  # (batch, seq, dim)
                norms = torch.norm(activations[key][0], dim=-1).mean().item()
                fig.add_trace(
                    go.Scatter(x=[key], y=[norms], mode='markers', 
                              name=f'{key} avg norm', marker_size=10),
                    row=1, col=2
                )
        
        # 3. Cross-attention 전후 비교
        if 'latent_init' in activations and 'cross_attention_0' in activations:
            before = activations['latent_init'].mean(dim=0).numpy()
            after = activations['cross_attention_0'][0].mean(dim=0).numpy()
            
            fig.add_trace(
                go.Scatter(y=before, mode='lines', name='Before Cross-Att', 
                          line=dict(color='blue')),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(y=after, mode='lines', name='After Cross-Att',
                          line=dict(color='red')),
                row=2, col=1
            )
        
        # 4. Self-attention을 통한 변화
        self_att_keys = [k for k in activations.keys() if 'self_attention' in k]
        for i, key in enumerate(sorted(self_att_keys)[:4]):  # 처음 4개만
            if activations[key].dim() == 3:
                avg_activation = activations[key][0].mean(dim=0).numpy()
                fig.add_trace(
                    go.Scatter(y=avg_activation, mode='lines', 
                              name=f'Self-Att Layer {i}',
                              opacity=0.7),
                    row=2, col=2
                )
        
        fig.update_layout(
            title_text="Attention Mechanisms Analysis",
            showlegend=True,
            height=800
        )
        
        if save_path:
            plot(fig, filename=save_path.replace('.png', '.html'), auto_open=False)
            print(f"Interactive attention plot saved to {save_path.replace('.png', '.html')}")
        
        fig.show()
    
    def plot_attention_maps(self, attention_weights: Dict, save_path: str = None):
        """어텐션 가중치 맵 시각화"""
        attention_keys = [k for k in attention_weights.keys() if 'weights' in k]
        
        if not attention_keys:
            print("No attention weights found!")
            return
        
        # 서브플롯 개수 계산
        n_maps = min(len(attention_keys), 8)  # 최대 8개까지
        rows = (n_maps + 3) // 4  # 4열로 배치
        cols = min(4, n_maps)
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        fig.suptitle('Attention Weight Maps', fontsize=16)
        
        for idx, key in enumerate(attention_keys[:n_maps]):
            row, col = idx // cols, idx % cols
            
            attention_map = attention_weights[key]  # (batch, num_heads, seq_q, seq_k)
            
            if attention_map.dim() == 4:
                # 첫 번째 배치, 첫 번째 헤드 사용
                att_map = attention_map[0, 0].numpy()
                
                # 어텐션 맵 그리기
                im = axes[row, col].imshow(att_map, cmap='Blues', aspect='auto')
                axes[row, col].set_title(f'{key.replace("_", " ").title()}')
                axes[row, col].set_xlabel('Key Position')
                axes[row, col].set_ylabel('Query Position')
                
                # 컬러바 추가
                plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
                
                # 높은 어텐션 값들 표시
                if att_map.shape[0] <= 20 and att_map.shape[1] <= 20:  # 작은 맵만
                    for i in range(att_map.shape[0]):
                        for j in range(att_map.shape[1]):
                            if att_map[i, j] > att_map.max() * 0.7:  # 상위 30% 값만
                                axes[row, col].text(j, i, f'{att_map[i, j]:.2f}', 
                                                  ha='center', va='center', fontsize=8)
        
        # 빈 서브플롯 숨기기
        for idx in range(n_maps, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention maps saved to {save_path}")
        plt.show()
    
    def plot_attention_head_comparison(self, attention_weights: Dict, layer_name: str, save_path: str = None):
        """특정 레이어의 멀티헤드 어텐션 비교"""
        if layer_name not in attention_weights:
            print(f"Layer {layer_name} not found in attention weights")
            return
        
        attention_map = attention_weights[layer_name]  # (batch, num_heads, seq_q, seq_k)
        
        if attention_map.dim() != 4:
            print(f"Invalid attention map shape: {attention_map.shape}")
            return
        
        batch_size, num_heads, seq_q, seq_k = attention_map.shape
        
        # 첫 번째 배치의 모든 헤드 시각화
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Multi-Head Attention: {layer_name.replace("_", " ").title()}', fontsize=16)
        
        for head in range(min(8, num_heads)):  # 최대 8개 헤드
            row, col = head // 4, head % 4
            
            att_map = attention_map[0, head].numpy()
            
            im = axes[row, col].imshow(att_map, cmap='Blues', aspect='auto')
            axes[row, col].set_title(f'Head {head + 1}')
            axes[row, col].set_xlabel('Key Position')
            axes[row, col].set_ylabel('Query Position')
            
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)
        
        # 빈 헤드 숨기기
        for head in range(min(8, num_heads), 8):
            row, col = head // 4, head % 4
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multi-head attention comparison saved to {save_path}")
        plt.show()
    
    def plot_information_flow(self, activations: Dict, save_path: str = None):
        """정보 흐름 시각화"""
        plt.figure(figsize=(15, 10))
        
        # 각 레이어에서의 활성화 통계 계산
        layer_stats = {}
        layer_order = []
        
        for key in activations.keys():
            if key == 'latent_init':
                data = activations[key]
            else:
                data = activations[key][0] if activations[key].dim() == 3 else activations[key]
            
            # 통계 계산
            mean_val = data.mean().item()
            std_val = data.std().item()
            max_val = data.max().item()
            min_val = data.min().item()
            
            layer_stats[key] = {
                'mean': mean_val,
                'std': std_val,
                'max': max_val,
                'min': min_val
            }
            layer_order.append(key)
        
        # 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Information Flow Through Perceiver Architecture', fontsize=16)
        
        # 평균 활성화 값
        means = [layer_stats[key]['mean'] for key in layer_order]
        axes[0, 0].plot(range(len(layer_order)), means, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Mean Activation Values')
        axes[0, 0].set_xticks(range(len(layer_order)))
        axes[0, 0].set_xticklabels([k.replace('_', '\n') for k in layer_order], rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 표준편차 (활성화 다양성)
        stds = [layer_stats[key]['std'] for key in layer_order]
        axes[0, 1].plot(range(len(layer_order)), stds, 'ro-', linewidth=2, markersize=8)
        axes[0, 1].set_title('Activation Diversity (Std Dev)')
        axes[0, 1].set_xticks(range(len(layer_order)))
        axes[0, 1].set_xticklabels([k.replace('_', '\n') for k in layer_order], rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 활성화 범위 (max - min)
        ranges = [layer_stats[key]['max'] - layer_stats[key]['min'] for key in layer_order]
        axes[1, 0].plot(range(len(layer_order)), ranges, 'go-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Activation Range (Max - Min)')
        axes[1, 0].set_xticks(range(len(layer_order)))
        axes[1, 0].set_xticklabels([k.replace('_', '\n') for k in layer_order], rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 정보량 변화 (엔트로피 근사)
        entropies = []
        for key in layer_order:
            if key == 'latent_init':
                data = activations[key].flatten()
            else:
                data = activations[key][0].flatten() if activations[key].dim() == 3 else activations[key].flatten()
            
            # 히스토그램 기반 엔트로피 계산
            hist, _ = np.histogram(data.numpy(), bins=50, density=True)
            hist = hist[hist > 0]  # 0이 아닌 값만
            entropy = -np.sum(hist * np.log(hist + 1e-10))
            entropies.append(entropy)
        
        axes[1, 1].plot(range(len(layer_order)), entropies, 'mo-', linewidth=2, markersize=8)
        axes[1, 1].set_title('Information Content (Entropy)')
        axes[1, 1].set_xticks(range(len(layer_order)))
        axes[1, 1].set_xticklabels([k.replace('_', '\n') for k in layer_order], rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Information flow plot saved to {save_path}")
        plt.show()
    
    def analyze_single_sample(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                             save_dir: str = './visualizations'):
        """단일 샘플에 대한 전체 분석"""
        os.makedirs(save_dir, exist_ok=True)
        
        print("Registering hooks...")
        self.register_hooks()
        
        print("Running forward pass...")
        output, activations, attention_weights = self.forward_with_visualization(x, mask)
        
        print("\nModel Output:")
        print(f"Prediction: {torch.argmax(output, dim=-1).item()}")
        print(f"Confidence: {torch.softmax(output, dim=-1).max().item():.4f}")
        
        print("\nActivation shapes:")
        for key, value in activations.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: {value.shape}")
        
        # 시각화 생성
        print("\nGenerating visualizations...")
        
        # 1. Latent evolution
        self.plot_latent_evolution(
            activations, 
            save_path=os.path.join(save_dir, 'latent_evolution.png')
        )
        
        # 2. Attention patterns
        self.plot_attention_patterns(
            activations,
            save_path=os.path.join(save_dir, 'attention_patterns.png')
        )
        
        # 3. Information flow
        self.plot_information_flow(
            activations,
            save_path=os.path.join(save_dir, 'information_flow.png')
        )
        
        # 4. Attention maps
        self.plot_attention_maps(
            attention_weights,
            save_path=os.path.join(save_dir, 'attention_maps.png')
        )
        
        # 5. Multi-head attention comparison (첫 번째 cross-attention)
        if 'cross_attention_weights_0' in attention_weights:
            self.plot_attention_head_comparison(
                attention_weights, 
                'cross_attention_weights_0',
                save_path=os.path.join(save_dir, 'multihead_attention.png')
            )
        
        # Hook 제거
        self.remove_hooks()
        
        return output, activations, attention_weights


def load_trained_model(checkpoint_path: str, device: str = 'cpu'):
    """학습된 모델을 체크포인트에서 로드"""
    print(f"Loading trained model from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 모델 생성
    model = create_perceiver_model(
        input_dim=config['input_dim'],
        num_classes=config['num_classes'], 
        model_size=config['model_size'],
        num_latents=config.get('num_latents', None)
    )
    
    # 가중치 로드
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully! Best accuracy: {checkpoint.get('best_accuracy', 'Unknown'):.4f}")
    return model, config

def demo_visualization(checkpoint_path: str = None):
    """시각화 데모"""
    print("=== Perceiver Model Visualization Demo ===")
    
    # 모델과 데이터 준비
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        # 학습된 모델 로드
        model, config = load_trained_model(checkpoint_path, device)
        dataset_type = config.get('dataset_type', 'synthetic')
        input_dim = config['input_dim']
        num_classes = config['num_classes']
    else:
        # 랜덤 초기화 모델 사용
        print("No checkpoint provided. Using randomly initialized model.")
        model = create_perceiver_model(
            input_dim=32, 
            num_classes=10, 
            model_size='base'
        )
        dataset_type = 'synthetic'
        input_dim = 32
        num_classes = 10
    
    # 데이터 생성
    if dataset_type == 'cifar10':
        from data_utils import create_cifar10_sequence_dataset
        dataloader = create_cifar10_sequence_dataset(
            batch_size=1,
            patch_size=8,
            train=False
        )
    else:
        dataloader = create_synthetic_sequence_dataloader(
            batch_size=1,  # 단일 샘플 분석
            num_samples=10,
            feature_dim=input_dim,
            num_classes=num_classes
        )
    
    # 시각화 도구 초기화
    visualizer = ModelVisualizer(model, device=device)
    
    # 첫 번째 샘플 분석
    for batch in dataloader:
        x = batch['input']
        mask = batch['mask']
        label = batch['label']
        
        print(f"\nAnalyzing sample with label: {label.item()}")
        print(f"Input shape: {x.shape}")
        
        # 전체 분석 실행
        output, activations, attention_weights = visualizer.analyze_single_sample(
            x, mask, save_dir='./visualizations'
        )
        
        break
    
    print("\nVisualization complete! Check the './visualizations' folder for results.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Perceiver 모델 시각화')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='학습된 모델 체크포인트 경로')
    
    args = parser.parse_args()
    
    demo_visualization(args.checkpoint)