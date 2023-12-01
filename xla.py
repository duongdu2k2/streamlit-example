import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import streamlit as st

def apply_negative(image):
    return 255 - image


def apply_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_threshold(image, threshold_value):
    _, thresh_binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh_binary


def apply_weighted_mean(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    result = cv2.filter2D(image, -1, kernel)
    return result


def apply_k_nearest_mean(image, k, threshold):
    result = np.zeros_like(image, dtype=np.float32)
    rows, cols = image.shape[:2]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = image[i - 1:i + 2, j - 1:j + 2].flatten()
            window = np.sort(window)
            window_size = len(window)

            # Tính trung bình cho k giá trị gần nhất
            mean_value = np.sum(window[:k]) / k

            # Kiểm tra ngưỡng cho từng phần tử trong cửa sổ
            condition = np.abs(image[i, j] - mean_value) > threshold
            if np.any(condition):
                result[i, j] = image[i, j]
            else:
                result[i, j] = mean_value

    return np.uint8(result)


def apply_median_filter(image, kernel_size):
    result = np.zeros_like(image)
    rows, cols = image.shape[:2]

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = image[i - 1:i + 2, j - 1:j + 2].flatten()
            median_value = np.median(window)
            result[i, j] = median_value

    return np.uint8(result)


def apply_histogram_equalization(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    lut = np.interp(image.flatten(), bins[:-1], cdf_normalized)
    result = np.uint8(lut.reshape(image.shape))

    return result


def apply_logarithmic_transformation(image, c=1):
    # Đảm bảo ảnh ở dạng số thực
    image = image.astype(np.float32) + 1.0

    # Áp dụng biến đổi logarithmic
    result = c * np.log1p(image)

    # Chuẩn hóa giá trị pixel về khoảng [0, 255]
    result = ((result - np.min(result)) / (np.max(result) - np.min(result))) * 255.0

    return np.uint8(result)


def apply_power_law_transformation(image, c, gamma):
    # Đảm bảo ảnh ở dạng số thực
    image = image.astype(np.float32) + 1.0

    # Áp dụng biến đổi hàm mũ
    result = c * np.power(image, gamma)

    # Chuẩn hóa giá trị pixel về khoảng [0, 255]
    result = ((result - np.min(result)) / (np.max(result) - np.min(result))) * 255.0

    return np.uint8(result)


def roberts_cross(image):
    # Kernel Roberts cho đạo hàm theo hướng ngang
    kernel_x = np.array([[1, 0], [0, -1]])

    # Kernel Roberts cho đạo hàm theo hướng dọc
    kernel_y = np.array([[0, 1], [-1, 0]])

    # Áp dụng các kernel để tính đạo hàm riêng theo hướng ngang và dọc
    vertical = cv2.filter2D(image, -1, kernel_x)
    horizontal = cv2.filter2D(image, -1, kernel_y)

    # Tính toán độ lớn của đạo hàm
    gradient_magnitude = np.sqrt(vertical ** 2 + horizontal ** 2)
    gradient_magnitude *= 255

    return np.uint8(gradient_magnitude)


def prewitt_operator(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Chuyển ảnh sang kiểu dữ liệu double
    gray_image_double = gray_image.astype(np.float64)

    # Kernel Prewitt cho đạo hàm theo hướng ngang
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    # Kernel Prewitt cho đạo hàm theo hướng dọc
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Áp dụng các kernel để tính toán đạo hàm theo hướng ngang và dọc
    gradient_x = cv2.filter2D(gray_image_double, -1, kernel_x)
    gradient_y = cv2.filter2D(gray_image_double, -1, kernel_y)

    # Tính toán độ lớn của đạo hàm toàn cục (gradiant magnitude)
    gradient_magnitude = (np.sqrt(gradient_x ** 2 + gradient_y ** 2))

    return np.uint8(gradient_magnitude)


def sobel_operator(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Chuyển ảnh sang kiểu dữ liệu double
    gray_image_double = gray_image.astype(np.float64)
    # Kernel Sobel cho đạo hàm theo hướng ngang
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    # Kernel Sobel cho đạo hàm theo hướng dọc
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # Áp dụng các kernel để tính toán đạo hàm theo hướng ngang và dọc
    gradient_x = cv2.filter2D(gray_image_double, -1, kernel_x)
    gradient_y = cv2.filter2D(gray_image_double, -1, kernel_y)

    # Tính toán độ lớn của đạo hàm toàn cục (gradiant magnitude)
    gradient_magnitude = (np.sqrt(gradient_x ** 2 + gradient_y ** 2))

    return np.uint8(gradient_magnitude)


def apply_laplacian(image):
    # Chuyển ảnh sang ảnh grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng toán tử Laplacian
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)

    return np.uint8(laplacian)


def apply_canny(image, low_threshold, high_threshold):
    # Chuyển ảnh sang ảnh grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng thuật toán Canny
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)

    return np.uint8(edges)


def apply_otsu(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Áp dụng thuật toán Otsu
    _, otsu_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return np.uint8(otsu_image)


def isodata_threshold(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Khởi tạo giá trị ngưỡng ban đầu
    threshold_value = np.mean(image)

    while True:
        # Phân loại ảnh dựa trên giá trị ngưỡng
        _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        # Tính giá trị trung bình của các pixel trong mỗi nhóm
        mean1 = np.mean(image[binary_image == 0])
        mean2 = np.mean(image[binary_image != 0])

        # Cập nhật giá trị ngưỡng
        new_threshold_value = (mean1 + mean2) / 2

        # Kiểm tra điều kiện dừng
        if np.abs(new_threshold_value - threshold_value) < 0.5:
            break

        threshold_value = new_threshold_value

    return np.uint8(binary_image)


def perform_erosion(image, kernel_size, iterations):
    # Đọc ảnh
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tạo kernel cho phép co
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Thực hiện phép co
    eroded_image = cv2.erode(gray_image, kernel, iterations)

    return eroded_image


def perform_dilation(image, kernel_size, iterations):
    # Đọc ảnh
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Tạo kernel cho phép co
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Thực hiện phép co
    eroded_image = cv2.dilate(gray_image, kernel, iterations)

    return eroded_image


def perform_opening(image, kernel_size):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    opened_image = cv2.morphologyEx(gray_image, cv2.MORPH_OPEN, kernel)

    return opened_image


def perform_closing(image, kernel_size):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_image = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)

    return closed_image


# Đường dẫn ảnh
image_paths = "/Users/duongquangdu/Downloads/doremon.jpg"

# Tạo danh sách các thuật toán
list_algorithm = [
    "Negative Filter",
    "Grayscale Filter",
    "Threshold Filter",
    "Weighted Mean Filter",
    "k-Nearest Mean Filter",
    "Median Filter",
    "Histogram Equalization",
    "Logarith",
    "Pow law",
    "Roberts cross",
    "Prewitt",
    "Sobel",
    "Laplacian",
    "Canny",
    "OTSU",
    "Isodata",
    "Erosion",
    "Dilation",
    "Opening",
    "Closing",
]

# Tạo danh sách mục đích các thuật toán
purposes = [
    "Âm bản hóa ảnh: Tạo ảnh đảo ngược để nổi bật các chi tiết.",
    "Chuyển đổi ảnh sang grayscale: Giảm chiều sâu màu và chuẩn bị cho các xử lý tiếp theo.",
    "Phân ngưỡng ảnh: Tạo ảnh nhị phân để tách vùng sáng và tối.",
    "Làm mờ ảnh bằng bộ lọc trung bình có trọng số: Làm mờ ảnh để giảm nhiễu và làm nổi bật đối tượng.",
    "Lọc ảnh bằng giá trị k giá trị gần nhất: Giảm nhiễu và làm mượt ảnh.",
    "Lọc ảnh bằng giá trị trung vị: Giảm nhiễu và làm mượt ảnh.",
    "Cân bằng lược đồ xám: Tăng độ tương phản và cân bằng độ sáng của ảnh.",
    "Biến đổi logarithmic: Điều chỉnh độ tương phản và chiếu sáng ảnh.",
    "Biến đổi hàm mũ: Điều chỉnh độ tương phản và chiếu sáng ảnh.",
    "Tính toán độ lớn của đạo hàm sử dụng toán tử Roberts: Phát hiện cạnh trong ảnh.",
    "Tính toán độ lớn của đạo hàm sử dụng toán tử Prewitt: Phát hiện cạnh trong ảnh.",
    "Tính toán độ lớn của đạo hàm sử dụng toán tử Sobel: Phát hiện cạnh trong ảnh.",
    "Áp dụng toán tử Laplacian: Làm nổi bật cạnh và chi tiết trong ảnh.",
    "Áp dụng thuật toán Canny: Phát hiện cạnh trong ảnh bằng thuật toán Canny.",
    "Áp dụng phân ngưỡng Otsu: Tự động xác định ngưỡng phân loại.",
    "Áp dụng phân ngưỡng Isodata: Xác định ngưỡng phân loại dựa trên giá trị trung bình.",
    "Thực hiện phép co: Giảm kích thước vùng sáng trong ảnh.",
    "Thực hiện phép giãn: Tăng kích thước vùng sáng trong ảnh.",
    "Thực hiện phép mở: Loại bỏ những vùng nhỏ và nối các vùng lớn.",
    "Thực hiện phép đóng: Loại bỏ các lỗ và kết hợp các vùng gần nhau."
]



st.title("Bài tập lớn: Xử lý ảnh")

original_image = cv2.imread(image_paths)

chosen_algorithm_name = st.sidebar.selectbox("Choose an algorithm:", list_algorithm)

if chosen_algorithm_name:
    st.subheader(f"Algorithm: {chosen_algorithm_name}")
    st.write(purposes[list_algorithm.index(chosen_algorithm_name)])
    
    if chosen_algorithm_name == "Negative Filter":
        result = apply_negative(original_image)
    elif chosen_algorithm_name == "Grayscale Filter":
        result = apply_grayscale(original_image)
    elif chosen_algorithm_name == "Threshold Filter":
        threshold_value = float(st.slider("Enter the threshold value: ", 0, 255))
        result = apply_threshold(original_image, threshold_value)
    elif chosen_algorithm_name == "Weighted Mean Filter":
        kernel_size = int(st.slider("Enter the kernel size (odd number): ", 1, 99))
        result = apply_weighted_mean(original_image, kernel_size)
    elif chosen_algorithm_name == "k-Nearest Mean Filter":
        k_value = int(st.slider("Enter the k value (odd number): ", 1, 8))
        threshold_value = int(st.slider("Enter the threshold value: ", 0, 255))
        result = apply_k_nearest_mean(original_image, k_value, threshold_value)
    elif chosen_algorithm_name == "Median Filter":
        kernel_size = int(st.slider("Enter the kernel size: ", 1, 99))
        result = apply_median_filter(original_image, kernel_size)
    elif chosen_algorithm_name == "Histogram Equalization":
        result = apply_histogram_equalization(original_image)
    elif chosen_algorithm_name == "Logarith":
        result = apply_logarithmic_transformation(original_image)
    elif chosen_algorithm_name == "Pow law":
        c_value = int(st.slider("Enter the c value: ", 1, 99))
        gamma_value = int(st.slider("Enter the gamma value: ", 1, 20))
        result = apply_power_law_transformation(original_image, c_value, gamma_value)
    elif chosen_algorithm_name == "Roberts cross":
        result = roberts_cross(original_image)
    elif chosen_algorithm_name == "Prewitt":
        result = prewitt_operator(original_image)
    elif chosen_algorithm_name == "Sobel":
        result = sobel_operator(original_image)
    elif chosen_algorithm_name == "Laplacian":
        result = apply_laplacian(original_image)
    elif chosen_algorithm_name == "Canny":
        low_threshold = int(st.slider("Enter the low_threshold value: ", 0, 255))
        high_threshold = int(st.slider("Enter the high_threshold value: ", low_threshold, 255))
        result = apply_canny(original_image, low_threshold, high_threshold)
    elif chosen_algorithm_name == "OTSU":
        result = apply_otsu(original_image)
    elif chosen_algorithm_name == "Isodata":
        result = isodata_threshold(original_image)
    elif chosen_algorithm_name == "Erosion":
        kernel_size = int(st.slider("Enter the kernel value: ", 1, 99))
        iterations = int(st.slider("Enter the iterations value: ", 0, 99))
        result = perform_erosion(original_image, kernel_size, iterations)
    elif chosen_algorithm_name == "Dilation":
        kernel_size = int(st.slider("Enter the kernel value: ", 1, 99))
        iterations = int(st.slider("Enter the iterations value: ", 0, 99))
        result = perform_dilation(original_image, kernel_size, iterations)
    elif chosen_algorithm_name == "Opening":
        kernel_size = int(st.slider("Enter the kernel value: ", 1, 99))
        result = perform_opening(original_image, kernel_size)
    elif chosen_algorithm_name == "Closing":
        kernel_size = int(st.slider("Enter the kernel value: ", 1, 99))
        result = perform_closing(original_image, kernel_size)

    col1, col2 = st.columns(2)

    col1.header("Original Image")
    col1.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), use_column_width=True)

    col2.header(f"Algorithm:")
    col2.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), use_column_width=True)