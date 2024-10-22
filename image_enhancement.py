import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_image(image_path):
    # Đọc ảnh màu
    image = cv2.imread(image_path)

    # Kiểm tra nếu ảnh được đọc thành công
    if image is None:
        print("Không thể đọc ảnh. Vui lòng kiểm tra đường dẫn.")
        return

    # Chuyển đổi từ BGR sang RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Tăng cường ảnh cho từng kênh màu
    channels = cv2.split(image)
    enhanced_channels = []

    # 1. Ảnh âm tính cho từng kênh màu
    negative_image = 255 - image

    # 2. Tăng cường độ tương phản (CLAHE) cho từng kênh màu
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for channel in channels:
        enhanced_channel = clahe.apply(channel)
        enhanced_channels.append(enhanced_channel)

    contrast_enhanced_image = cv2.merge(enhanced_channels)

    # 3. Biến đổi log cho từng kênh màu
    log_transformed_channels = []
    for channel in channels:
        c = 255 / np.log(1 + np.max(channel))
        log_transformed_image = c * np.log(1 + channel)
        log_transformed_channels.append(np.array(log_transformed_image, dtype=np.uint8))

    log_transformed_image = cv2.merge(log_transformed_channels)

    # 4. Cân bằng histogram cho từng kênh màu
    histogram_equalized_channels = []
    for channel in channels:
        histogram_equalized_channels.append(cv2.equalizeHist(channel))

    histogram_equalized_image = cv2.merge(histogram_equalized_channels)

    # Hiển thị kết quả
    plt.figure(figsize=(12, 10))

    # Subplot cho từng ảnh
    titles = ['Ảnh gốc', 'Ảnh âm tính', 'Tăng cường độ tương phản', 'Biến đổi log', 'Cân bằng histogram']
    images = [image, negative_image, contrast_enhanced_image, log_transformed_image, histogram_equalized_image]

    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[i])
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Thay đổi đường dẫn đến ảnh của bạn tại đây
if __name__ == "__main__":
    image_path = 'istockphoto-517188688-612x612.jpg'  # Thay đổi đường dẫn
    enhance_image(image_path)
