def process_patches(batch_images, patch_size=8):
    """
    Args:
        batch_images (Tensor): CPU 상의 이미지 배치 [B, 1, H, W]
        patch_size (int): 패치 크기 (기본 8)
    Returns:
        shuffled_patches (Tensor): GPU 상의 셔플된 패치들 [Total_Patches, 1, 8, 8]
    """
    # 1. GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_gpu = batch_images.to(device) # Shape: [B, 1, 256, 256]
    
    B, C, H, W = img_gpu.shape
    
    # 2. Unfold를 사용하여 겹치지 않게 패치 추출
    # stride=patch_size 로 설정하여 겹치지 않게 함
    # unfold(dimension, size, step)
    # 결과 Shape: [B, C, H_steps, W_steps, patch_h, patch_w]
    # 256 / 8 = 32 이므로 H_steps, W_steps는 32가 됨
    patches = img_gpu.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    
    # 3. 차원 정리 (Flatten to Batch)
    # 현재: [B, 1, 32, 32, 8, 8]
    # 목표: [B * 32 * 32, 1, 8, 8]
    
    # (1) 먼저 패치 개수 차원들을 배치 쪽으로 몰아주기 위해 permute
    # [B, C, H_n, W_n, ph, pw] -> [B, H_n, W_n, C, ph, pw]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    
    # (2) View를 통해 하나의 큰 배치로 병합
    # -1은 자동으로 계산된 총 패치 수 (B * 32 * 32)
    patches = patches.view(-1, C, patch_size, patch_size)
    
    # 4. Batch 축을 랜덤하게 섞기 (Shuffle)
    # 총 패치 개수만큼의 랜덤 인덱스 생성
    total_patches = patches.size(0)
    rand_idx = torch.randperm(total_patches).to(device)
    
    shuffled_patches = patches[rand_idx]
    
    return shuffled_patches

# --- 실행 예시 ---
if __name__ == '__main__':
    # 1. 더미 데이터 로더 생성 (위의 데이터셋 클래스 사용 가정)
    # root_dir은 실제 경로로 수정 필요
    dataset = YDomainImageDataset(root_dir='./data/train', mode='train', crop_size=256)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 2. 배치 하나 가져오기
    try:
        images = next(iter(loader)) # [4, 1, 256, 256]
        print(f"Original Batch Shape: {images.shape}")

        # 3. 패치 처리 함수 호출
        final_patches = process_patches(images, patch_size=8)
        
        print(f"Processed Patches Shape: {final_patches.shape}")
        # 예상 결과 계산:
        # 이미지 1장당 (256/8)*(256/8) = 32*32 = 1024개 패치
        # 배치가 4장이므로 4 * 1024 = 4096
        # 최종 Shape: [4096, 1, 8, 8] (순서는 뒤죽박죽 섞임)
        
    except Exception as e:
        print(f"데이터 로딩 실패 (경로 확인 필요): {e}")
