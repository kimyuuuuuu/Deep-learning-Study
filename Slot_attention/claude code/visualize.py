import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
import os
from PIL import Image
import seaborn as sns


def visualize_slots(original_images, reconstructed_images, masks, save_path=None, figsize=(15, 10)):
    """
    슬롯 어텐션 결과를 시각화합니다.
    
    Args:
        original_images: (N, 3, H, W) - 원본 이미지들
        reconstructed_images: (N, 3, H, W) - 재구성된 이미지들  
        masks: (N, num_slots, H, W) - 각 슬롯의 어텐션 마스크들
        save_path: 저장할 경로 (선택적)
        figsize: 그림 크기
        
    Returns:
        matplotlib figure 객체
    """
    N = original_images.size(0)
    num_slots = masks.size(1)
    
    # 텐서를 numpy로 변환하고 올바른 형태로 변형
    original_np = original_images.permute(0, 2, 3, 1).numpy()
    reconstructed_np = reconstructed_images.permute(0, 2, 3, 1).numpy()
    masks_np = masks.numpy()
    
    # 값 범위를 [0, 1]로 클리핑
    original_np = np.clip(original_np, 0, 1)
    reconstructed_np = np.clip(reconstructed_np, 0, 1)
    
    # 그림 생성: 원본, 재구성, 각 슬롯 마스크들
    cols = 2 + num_slots  # 원본 + 재구성 + 슬롯들
    fig, axes = plt.subplots(N, cols, figsize=figsize)
    
    if N == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(N):
        # 원본 이미지
        axes[i, 0].imshow(original_np[i])
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')
        
        # 재구성된 이미지
        axes[i, 1].imshow(reconstructed_np[i])
        axes[i, 1].set_title('Reconstructed')
        axes[i, 1].axis('off')
        
        # 각 슬롯의 마스크들
        for slot_idx in range(num_slots):
            mask = masks_np[i, slot_idx]
            axes[i, 2 + slot_idx].imshow(mask, cmap='Blues', vmin=0, vmax=1)
            axes[i, 2 + slot_idx].set_title(f'Slot {slot_idx}')
            axes[i, 2 + slot_idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"시각화 결과 저장: {save_path}")
    
    return fig


def visualize_attention_evolution(attention_weights, save_path=None, figsize=(20, 5)):
    """
    슬롯 어텐션의 반복별 진화 과정을 시각화합니다.
    
    Args:
        attention_weights: (B, num_iterations, H*W, num_slots) - 어텐션 가중치들
        save_path: 저장 경로 (선택적)
        figsize: 그림 크기
        
    Returns:
        matplotlib figure 객체
    """
    batch_idx = 0  # 첫 번째 배치만 시각화
    num_iterations, num_pixels, num_slots = attention_weights.shape[1:4]
    
    # 이미지 해상도 추정 (정사각형 가정)
    resolution = int(np.sqrt(num_pixels))
    
    # 어텐션 가중치를 이미지 형태로 변형
    attn_np = attention_weights[batch_idx].cpu().numpy()  # (num_iterations, H*W, num_slots)
    attn_reshaped = attn_np.reshape(num_iterations, resolution, resolution, num_slots)
    
    fig, axes = plt.subplots(num_slots, num_iterations, figsize=figsize)
    
    if num_slots == 1:
        axes = axes.reshape(1, -1)
    if num_iterations == 1:
        axes = axes.reshape(-1, 1)
    
    for slot_idx in range(num_slots):
        for iter_idx in range(num_iterations):
            attention_map = attn_reshaped[iter_idx, :, :, slot_idx]
            
            im = axes[slot_idx, iter_idx].imshow(attention_map, cmap='viridis', vmin=0, vmax=1)
            axes[slot_idx, iter_idx].set_title(f'Slot {slot_idx}, Iter {iter_idx+1}')
            axes[slot_idx, iter_idx].axis('off')
    
    # 컬러바 추가
    plt.colorbar(im, ax=axes, shrink=0.8, label='Attention Weight')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"어텐션 진화 과정 저장: {save_path}")
    
    return fig


def visualize_slot_decomposition(original_image, reconstructed_image, masks, 
                               slot_reconstructions=None, save_path=None, figsize=(20, 4)):
    """
    하나의 이미지에 대한 슬롯 분해 과정을 상세히 시각화합니다.
    
    Args:
        original_image: (3, H, W) - 원본 이미지
        reconstructed_image: (3, H, W) - 재구성된 이미지
        masks: (num_slots, H, W) - 슬롯별 마스크들
        slot_reconstructions: (num_slots, 3, H, W) - 슬롯별 재구성 결과 (선택적)
        save_path: 저장 경로 (선택적)
        figsize: 그림 크기
        
    Returns:
        matplotlib figure 객체
    """
    num_slots = masks.size(0)
    
    # 텐서를 numpy로 변환
    original_np = original_image.permute(1, 2, 0).cpu().numpy()
    reconstructed_np = reconstructed_image.permute(1, 2, 0).cpu().numpy()
    masks_np = masks.cpu().numpy()
    
    original_np = np.clip(original_np, 0, 1)
    reconstructed_np = np.clip(reconstructed_np, 0, 1)
    
    # 서브플롯 구성
    if slot_reconstructions is not None:
        cols = 2 + 2 * num_slots  # 원본 + 재구성 + (마스크 + 슬롯재구성) * num_slots
        slot_recons_np = slot_reconstructions.permute(0, 2, 3, 1).cpu().numpy()
        slot_recons_np = np.clip(slot_recons_np, 0, 1)
    else:
        cols = 2 + num_slots  # 원본 + 재구성 + 마스크들
    
    fig, axes = plt.subplots(1, cols, figsize=figsize)
    
    col_idx = 0
    
    # 원본 이미지
    axes[col_idx].imshow(original_np)
    axes[col_idx].set_title('Original', fontsize=12, fontweight='bold')
    axes[col_idx].axis('off')
    col_idx += 1
    
    # 재구성된 이미지
    axes[col_idx].imshow(reconstructed_np)
    axes[col_idx].set_title('Reconstructed', fontsize=12, fontweight='bold')
    axes[col_idx].axis('off')
    col_idx += 1
    
    # 각 슬롯
    for slot_idx in range(num_slots):
        # 마스크 시각화
        mask = masks_np[slot_idx]
        axes[col_idx].imshow(mask, cmap='Blues', vmin=0, vmax=1)
        axes[col_idx].set_title(f'Slot {slot_idx} Mask', fontsize=10)
        axes[col_idx].axis('off')
        col_idx += 1
        
        # 슬롯별 재구성 결과 (있는 경우)
        if slot_reconstructions is not None:
            slot_recon = slot_recons_np[slot_idx]
            axes[col_idx].imshow(slot_recon)
            axes[col_idx].set_title(f'Slot {slot_idx} Recon', fontsize=10)
            axes[col_idx].axis('off')
            col_idx += 1
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"슬롯 분해 결과 저장: {save_path}")
    
    return fig


def create_attention_heatmap(attention_weights, original_image, save_path=None, alpha=0.6):
    """
    어텐션 가중치를 원본 이미지 위에 히트맵으로 오버레이합니다.
    
    Args:
        attention_weights: (num_slots, H, W) - 어텐션 가중치들
        original_image: (3, H, W) - 원본 이미지
        save_path: 저장 경로 (선택적)
        alpha: 히트맵 투명도
        
    Returns:
        matplotlib figure 객체
    """
    num_slots = attention_weights.size(0)
    
    # 데이터 변환
    original_np = original_image.permute(1, 2, 0).cpu().numpy()
    attn_np = attention_weights.cpu().numpy()
    
    original_np = np.clip(original_np, 0, 1)
    
    # 색상 팔레트 설정
    colors = plt.cm.Set1(np.linspace(0, 1, num_slots))
    
    fig, axes = plt.subplots(1, num_slots + 1, figsize=(4 * (num_slots + 1), 4))
    
    if num_slots == 0:
        axes = [axes]
    
    # 원본 이미지
    axes[0].imshow(original_np)
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')
    
    # 각 슬롯의 어텐션 히트맵
    for slot_idx in range(num_slots):
        axes[slot_idx + 1].imshow(original_np)
        
        # 어텐션 히트맵 오버레이
        attention_map = attn_np[slot_idx]
        im = axes[slot_idx + 1].imshow(attention_map, cmap='Reds', alpha=alpha, vmin=0, vmax=1)
        
        axes[slot_idx + 1].set_title(f'Slot {slot_idx} Attention', fontweight='bold')
        axes[slot_idx + 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"어텐션 히트맵 저장: {save_path}")
    
    return fig


def save_reconstruction_examples(model, dataloader, save_dir, num_examples=8, device='cpu'):
    """
    모델의 재구성 결과 예시들을 저장합니다.
    
    Args:
        model: 훈련된 슬롯 어텐션 모델
        dataloader: 데이터로더
        save_dir: 저장할 디렉토리
        num_examples: 저장할 예시 개수
        device: 실행 디바이스
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    examples_saved = 0
    
    with torch.no_grad():
        for batch_idx, images in enumerate(dataloader):
            if examples_saved >= num_examples:
                break
                
            images = images.to(device)
            recons, masks, slots, attn = model(images)
            
            batch_size = images.size(0)
            for i in range(min(batch_size, num_examples - examples_saved)):
                # 개별 이미지 처리
                original = images[i:i+1]
                reconstructed = recons[i:i+1] 
                mask = masks[i:i+1]
                
                # 시각화 및 저장
                save_path = os.path.join(save_dir, f'example_{examples_saved + 1}.png')
                fig = visualize_slots(original, reconstructed, mask, save_path=save_path)
                plt.close(fig)
                
                # 개별 슬롯별 상세 시각화
                detail_save_path = os.path.join(save_dir, f'detail_example_{examples_saved + 1}.png')
                fig = visualize_slot_decomposition(
                    images[i], recons[i], masks[i], save_path=detail_save_path
                )
                plt.close(fig)
                
                examples_saved += 1
                
                if examples_saved >= num_examples:
                    break
    
    print(f"{examples_saved}개의 재구성 예시를 {save_dir}에 저장했습니다.")


def plot_training_progress(train_losses, val_losses, save_path=None):
    """
    훈련 진행 상황을 시각화합니다.
    
    Args:
        train_losses: 훈련 손실 리스트
        val_losses: 검증 손실 리스트
        save_path: 저장 경로 (선택적)
        
    Returns:
        matplotlib figure 객체
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12) 
    ax.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 최솟값 표시
    min_train_epoch = np.argmin(train_losses) + 1
    min_val_epoch = np.argmin(val_losses) + 1
    
    ax.scatter([min_train_epoch], [min(train_losses)], color='blue', s=100, zorder=5)
    ax.scatter([min_val_epoch], [min(val_losses)], color='red', s=100, zorder=5)
    
    ax.annotate(f'Min Train: {min(train_losses):.4f}', 
                xy=(min_train_epoch, min(train_losses)),
                xytext=(min_train_epoch + len(epochs)*0.1, min(train_losses) + max(train_losses)*0.05),
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    ax.annotate(f'Min Val: {min(val_losses):.4f}', 
                xy=(min_val_epoch, min(val_losses)),
                xytext=(min_val_epoch + len(epochs)*0.1, min(val_losses) + max(val_losses)*0.1),
                arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"훈련 진행 곡선 저장: {save_path}")
    
    return fig


if __name__ == "__main__":
    # 시각화 테스트 코드
    print("시각화 모듈 테스트...")
    
    # 더미 데이터 생성
    batch_size, num_slots, height, width = 2, 4, 64, 64
    
    original = torch.rand(batch_size, 3, height, width)
    reconstructed = torch.rand(batch_size, 3, height, width) 
    masks = torch.rand(batch_size, num_slots, height, width)
    attention = torch.rand(batch_size, 3, height*width, num_slots)
    
    # 시각화 테스트
    print("슬롯 시각화 테스트...")
    fig1 = visualize_slots(original, reconstructed, masks)
    plt.show()
    
    print("어텐션 진화 시각화 테스트...")
    fig2 = visualize_attention_evolution(attention)
    plt.show()
    
    print("상세 분해 시각화 테스트...")
    fig3 = visualize_slot_decomposition(original[0], reconstructed[0], masks[0])
    plt.show()
    
    print("테스트 완료!")