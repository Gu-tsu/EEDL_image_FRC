from HEAD import *

class Config:
    def __init__(self):
#########################################################
        # 功能列表
        self.fusion_world = 0           # 核聚变装置展示
        self.data_compare = 0           # 两个文件对比差异
        self.data_discrete = 0          # 数据离散程度分析，使用file_path
        self.check_missing_files = 0    # 检查缺失文件，使用file_path
        self.file_extreme = 0           # 单文件极值，使用input_path
        self.path_extreme = 0           # 路径全局极值，使用file_path
        self.ini_plus_per = 1           # 初始值累加
#########################################################
        # 通用参数
        self.input_path = r"\\192.168.0.42\zhangsiyu_Linux\WD\per_data4.0\per8两周期1k间隔\initial_2D\current0_0th.txt"
        self.file_path = r"\\192.168.0.42\zhangsiyu_Linux\WD\per_data4.0\per8两周期1k间隔\magnetic_br"
        self.output_path = r'C:\Users\Administrator\Desktop\\' 
#########################################################
        # 文件对比参数
        self.file1 = r"\\192.168.0.42\zhangsiyu_Linux\WD\per_data4.0\per8两周期1k间隔\initial_2D\current0_0th.txt"
        self.file2 = r"\\192.168.0.42\zhangsiyu_Linux\WD\per_data4.0\per8两周期1k间隔\initial_2D\density_old_0th.txt"
        self.eps_threshold = 0.0001     # 灵敏度阈值
#########################################################
        # 缺失文件检查参数
        self.start_file = 1000           # 起始文件编号
        self.end_file = 700000          # 结束文件编号
        self.interval = 1000             # 间隔数
#########################################################
        # 初始值累加参数
        self.initial_file = r"\\192.168.0.42\zhangsiyu_Linux\WD\per_data4.0\per8两周期1k间隔\current_jr\001000.txt"
        self.nr, self.nth, self.nz = 129, 8, 321    # 二维模式下只需修改NR和nz
        self.dimension_choice = 3                  # 2D/3D
#########################################################

def compare_two_files(file1, file2, eps=0.01):
    with open(file1, 'r') as f:
        data_str1 = f.read().replace('D', 'E')
    arr1 = np.loadtxt(StringIO(data_str1))

    with open(file2, 'r') as f:
        data_str2 = f.read().replace('D', 'E')
    arr2 = np.loadtxt(StringIO(data_str2))

    print("arr1.shape =", arr1.shape)
    print("arr2.shape =", arr2.shape)
    if arr1.shape != arr2.shape:
        raise ValueError("!! The two arrays are not the same shape and cannot be compared")
    point_diff = np.zeros_like(arr1, dtype=float)       # 创建一个与 arr1/arr2 同形状的差异数组 point_diff

    # 如果两边都为 0 -> 差异 = 0
    both_zero_mask = (arr1 == 0) & (arr2 == 0)
    point_diff[both_zero_mask] = 0

    # 如果一边是 0，另一边不是 0 -> 差异 = 非零数的绝对值
    one_zero_arr1 = (arr1 == 0) & (arr2 != 0)
    point_diff[one_zero_arr1] = np.abs(arr2[one_zero_arr1])
    one_zero_arr2 = (arr2 == 0) & (arr1 != 0)
    point_diff[one_zero_arr2] = np.abs(arr1[one_zero_arr2])

    # 如果两边都非 0 -> 差异 = |a - b| / |a|
    both_nonzero_mask = (arr1 != 0) & (arr2 != 0)
    point_diff[both_nonzero_mask] = (
        np.abs(arr1[both_nonzero_mask] - arr2[both_nonzero_mask]) 
        / np.abs(arr1[both_nonzero_mask])
    )

    avg_diff = np.mean(point_diff)
    max_diff = np.max(point_diff)
    within_eps_ratio = np.sum(point_diff >= eps) / point_diff.size
    print(f"Average difference: {avg_diff:.6f}")
    print(f"Maximum difference: {max_diff:.6f}")
    print(f"Over-threshold ratio: {within_eps_ratio * 100:.3f}%")

def analyze_single_file(file_path):
    with open(file_path, 'r') as f:
        data_str = f.read().replace('D', 'E')
        data = np.loadtxt(StringIO(data_str))

    if data.ndim == 1:
        rows = 1
        cols = data.size
    else:
        rows, cols = data.shape
    total_elements = data.size
    print(f"data structure: ({rows}, {cols}) \nNumber of elements: {total_elements}")
    flat_data = data.flatten()      # 将数据展开为一维数组
    variance = np.var(flat_data)
    std_dev = np.std(flat_data)
    print(f"Variance(方差): {variance:.6f}")
    print(f"Standard deviation(标准差）: {std_dev:.6f}")

    plt.figure(figsize=(20, 5))
    plt.plot(flat_data, marker='o', markersize=1, linestyle='-', color='b')
    plt.xlabel('indexing')
    plt.ylabel('value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return variance, std_dev

def check_missing_files(start, end, interval, folder):
    expected_files = [f"{i:06d}.txt" for i in range(start, end + 1, interval)]      # 生成预期的文件名列表
    existing_files = set(os.listdir(folder))                                        # 获取现有的文件列表
    missing_files = [file for file in expected_files if file not in existing_files]
    print(f"Ideal number of files: {len(expected_files)}")
    print(f"Actual number of files: {len(existing_files)}")
    if missing_files:
        print("miss file: ")
        for file in missing_files:
            print(file)
    else:
        print("------\nNo missing files")

def sub_file_extrema(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = [
        float(value.replace('D', 'E'))
        for line in lines if line.strip()
        for value in line.strip().split()
    ]
    if not data:
        print("File has no valid numeric data")
        sys.exit(1)

    minimum = min(data)
    maximum = max(data)
    print(f"Minimum: {minimum} \nMaximum: {maximum}")

def sub_path_extreme(pathname):
    # 获取文件夹内所有文件并按扩展名（区分大小写）分组
    files_by_ext = {}
    for filename in os.listdir(pathname):
        file_path = os.path.join(pathname, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            if ext:
                ext = ext[1:]  # 去除点号
                if ext not in files_by_ext:
                    files_by_ext[ext] = []
                files_by_ext[ext].append(filename)

    # 遍历每种扩展名的文件组
    for ext, files in files_by_ext.items():
        if len(files) > 1:
            global_min = float('inf')   # 初始化全局最小值为正无穷
            global_max = float('-inf')  # 最大值为负无穷
            total_files = len(files)

            for i, filename in enumerate(files, 1):
                file_path = os.path.join(pathname, filename)
                sys.stdout.write(f"\rProcessing '{ext}' {i}/{total_files} : {filename}")
                sys.stdout.flush()
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    data = [
                        float(value.replace('D', 'E'))
                        for line in lines if line.strip()
                        for value in line.strip().split()
                    ]
                    if not data:
                        print(f"\n!! File: '{filename}' has no valid numeric data")
                        continue
                    file_min = min(data)
                    file_max = max(data)

                    # 更新全局最小值和最大值
                    global_min = min(global_min, file_min)
                    global_max = max(global_max, file_max)

                except ValueError as e:
                    print(f"\n!! File: '{filename}' contains non-numeric data: {e}")
                except Exception as e:
                    print(f"\n!!Exception occurred in the processing of file '{filename}' : {e}")

            print("------------")
            # 检查是否有有效数据
            if global_min != float('inf') and global_max != float('-inf'):
                print(f"Global Maximum: {global_max}")
                print(f"Global Minimum: {global_min}")
            else:
                print(f"\n'{ext}' No valid numerical data in the file to calculate the extreme values")
        elif len(files) == 1:
            print(f"'{ext}' Only one file, please use the single file processing function: {files[0]}")

def sub_ini_plus_per(config):
    # 前序检查
    if not os.path.isfile(config.initial_file):
        print(f"No initial file. '{config.initial_file}'。")
        sys.exit(1)
    if not os.path.isdir(config.file_path):
        print(f"Path folder not found")
        sys.exit(1)
    file_extension = '.txt'     # 文件后缀
    input_files = sorted([f for f in os.listdir(config.file_path) if f.endswith(file_extension)])
    if not input_files:
        print(f"No {file_extension} files in the path folder")
        sys.exit(1)
    if not os.path.isdir(config.output_path):
        print("Output path not found")
        sys.exit(1)
    else:
        # 生成一个随机命名的新文件夹
        random_suffix = ''.join([str(random.randint(0, 9)) for _ in range(5)])
        new_output_path = os.path.join(config.output_path, f"output_{random_suffix}")
        os.makedirs(new_output_path, exist_ok=True)
        print(f"Output directory has been created:\n{new_output_path}")

    # 验证数组
    if config.dimension_choice == 3:
        print("3D mode")
        shape = (config.nr, config.nth, config.nz)
        total_elements = shape[0] * shape[1] * shape[2]
    elif config.dimension_choice == 2:
        print("2D mode")
        shape = (config.nr, config.nz)
        total_elements = shape[0] * shape[1]
    else:
        print("Dimension must 2 or 3")
        sys.exit(1)
    print(f"Customized array shape: {shape}")

    # 读取初始文件的数据并转换
    with open(config.initial_file, 'r') as f:
        lines = f.readlines()
    initial_data = np.array([
        float(value.replace('D', 'E'))
        for line in lines if line.strip()
        for value in line.strip().split()
    ])

    # 检查数组形状
    if initial_data.size != total_elements:
        print(f"Initial file does not match the shape of the input path file array")
        sys.exit(1)
    print("Array shape matching success")

    # 遍历输入文件夹中的所有文件
    total_files = len(input_files)
    print(f"Number of files: {total_files}")
    print("---------------")
    for idx, input_file in enumerate(input_files, start=1):
        input_file_path = os.path.join(config.file_path, input_file)
        with open(input_file_path, 'r') as f:
            lines = f.readlines()
        input_data = np.array([
            float(value.replace('D', 'E'))
            for line in lines if line.strip()
            for value in line.strip().split()
        ])
        if input_data.size != initial_data.size:
            print(f"File: {input_file} Shape does not match the initial file, skip and continue")
            continue
        result_data = initial_data + input_data     # 逐元素相加
        output_file_path = os.path.join(new_output_path, input_file)    # 写入结果文件
        with open(output_file_path, 'w') as f:
            if config.dimension_choice == 3:
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        start_idx = i * shape[1] * shape[2] + j * shape[2]
                        end_idx = start_idx + shape[2]
                        line = " ".join(f"{num:22.15E}" for num in result_data[start_idx:end_idx])
                        f.write(line + '\n')  # 保持Fortran输出文件格式
            else:  # 2D
                for i in range(shape[0]):
                    line = " ".join(f"{num:22.15E}" for num in result_data[i * shape[1]:(i + 1) * shape[1]])
                    f.write(line + '\n')  # 每行对应y方向的值
        sys.stdout.write(f"\r Process: {idx}/{total_files} file: {input_file}")
        sys.stdout.flush()
    print()


def main():
    config = Config()
    print("###############################")
    variables = [
        # 功能封装列表
        config.data_compare, config.data_discrete, config.check_missing_files, 
        config.fusion_world, config.file_extreme, config.path_extreme, 
        config.ini_plus_per
    ]
    ones_count = sum(variables)
    if any(v not in (0, 1) for v in variables):
        print("Please select the correct function")
        sys.exit(1)
    elif ones_count > 1:
        print("Prevent path conflicts, resulting in data loss, not recommended to turn on more than one function")
        sys.exit(1)
    elif ones_count == 0:
        print("Please select a function")
        sys.exit(1)
    else:
        if config.fusion_world:
            print("展示核聚变装置：")
            webbrowser.open_new_tab("https://remdelaportemathurin.github.io/fusion-world/")
        if config.data_compare:
            print("开始文件对比：")
            compare_two_files(config.file1, config.file2, config.eps_threshold)
        if config.data_discrete:
            print("开始离散分析：")
            analyze_single_file(config.input_path)
        if config.check_missing_files:
            print("检查缺失文件：")
            check_missing_files(config.start_file, config.end_file, config.interval, config.file_path)
        if config.file_extreme:
            print("单文件极值分析：")
            sub_file_extrema(config.input_path)
        if config.path_extreme:
            print("路径全局极值分析：")
            sub_path_extreme(config.file_path)
        if config.ini_plus_per:
            print("执行初始值累加：")
            sub_ini_plus_per(config)

if __name__ == "__main__":
    main()
