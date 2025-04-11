from HEAD import *

class Config:
    def __init__(self):
######################################################
        # 功能列表
        self.dimension_2D = 0            # 二维平面数组文件显示图像
        self.dimension_3D = 0            # 三维立体数组文件显示图像
        self.GIF_2D = 0                  # 二维平面数组输出GIF
        self.GIF_3D = 0                  # 三维立体数组输出GIF
        self.single_nz_nth = 0           # 显示选择nz或nth数据折线图
        self.single_nr = 0               # 显示选择nr数据折线图
        self.point_Selection = 1         # 选取指定点的数据随时间演化
######################################################
        # 通用参数
        self.input_path = r"D:\ZHANGSIYU\M1_线性Hall-MHD扰动场FAT-CM\3.平衡状态下的3D扰动场的波动传播\平衡量初始值（321nz）\initial_2D\density0_0th.txt"
        self.file_path = r"D:\ZHANGSIYU\M1_线性Hall-MHD扰动场FAT-CM\2.平衡状态FRC场\结果数据\out"
        self.output_path = r'C:\Users\Administrator\Desktop\\'           # 尽量不修改, 部分会附带随机数避免覆盖
        # ------------------------
        self.semicircular = 0                        # 1: 输入半圆, 0: 输入全圆
        self.coordinate_system = 0                   # 1: 输入笛卡尔坐标, 0: 圆柱坐标
        self.NX, self.NY, self.NZ = 129, 129, 321           # 笛卡尔（初始文件）
        self.NR, self.nth, self.nz = 129, 8, 321            # 圆柱
        self.nr_up, self.nth_up, self.nz_up = 129, 360, 21  # 角度扩展
        # ------------------------
        self.Grid_Ratio = 1                          # 1: 装置比例, 0: 网格比例
        self.physical_R = 0.3857                     # 装置半径（m）0.3857
        self.physical_Z = 2.24                       # 装置轴向长度
######################################################
        # dimension_2D参数，自动检测网格尺寸无需设置，支持1/2折线数据
        self.File_save = 0                           # 是否保存文件：1 保存，0 不保存
        self.cylindrical = 0                         # 圆柱坐标系（用于轴向视角）
        self.white_mask = 1                          # 圆柱坐标系下圆形外区域掩模
        self.output_all_files = 0                    # 批量处理开关（默认无plot显示，自动保存）
        self.log_scale = 0                           # 是否使用对数尺度放大差异
        self.xlabel = 'r [m]'                        # x 轴标签
        self.ylabel = 'ε_B'                       # y 轴标签
        self.title = ''                              # 图像标题
        self.colorbar_label = 'ψ [Wb]'                # 颜色条标签n [/m^3]
        self.Line_1 = '(j × B)r'                     # 双折现模式下的线名
        self.Line_2 = '(∇ P)r'
######################################################
        # dimension_3D参数
        self.view_option = 3                         # 1: 轴向视图, 2: 径向视图, 其他: 装置直观视图
        self.transparency = 1                        # 透明度: 1:低透，2:中透，3:高透
        self.enable_zero = 0                         # 将值为0的数据点设为透明
        self.data_range = 1                          # 1: 颜色范围基于文件最小值和最大值, 0: 自定义范围
        self.vmin = 0.0                              # 数据范围
        self.vmax = 6.04049985068545e-06
######################################################
        # GIF_2D参数
        self.FPS = 10                                # GIF每秒帧数
        self.cylindrical_g = 0                       # 1: 圆柱视角  0: 笛卡尔视角
        self.log_scale_g = 0                         # 是否使用对数尺度放大差异
        self.xlabel_g = 'z[m](2.24)'                 # x 轴标签
        self.ylabel_g = 'r[m]*2(0.7714)'             # y 轴标签
        self.colorbar_label_g = 'n [/m^3]'           # 颜色条标签
        self.use_manual_range = 0                    # 1: 手动设置全局范围, 0: 动态范围
        self.manual_min = -3.90490252536991e-07      # 数据范围
        self.manual_max = 7.4554665005423e-07
######################################################
        # GIF_3D参数（不支持角度修改，不支持装置比例）
        self.fps = 5                                 # GIF每秒帧数
        self.USE_GLOBAL_RANGE = 0                    # 1：自定义全局范围，0：动态范围
        self.GLOBAL_MIN = 0.0
        self.GLOBAL_MAX = 3.04769635813648e-06
        self.transparency_gif = 1                    # 透明度: 1:低透，2:高透
######################################################
        # single_nz_nth参数（图像不保存）
        self.choice_nz_nth = 161                     # 选用要显示的nz位置或nth位置
        self.cylindrical_singel = 0                  # 1: 圆柱视角  0: 笛卡尔视角
        self.ylabel_cnznth = 'β'                     # 数据单位
        self.bulk_nz_nth = 0                         # 批处理
        self.title_clear = 1                         # 隐藏标题, 与single_nr功能共用>>>
######################################################
        # single_nr参数（图像不保存）
        self.choice_nr = 157                         # 选用要显示的nr位置(218/257,90/129位置+0.268m, 157/257,29/129位置+0.084m)
        self.cylindrical_singel_nr = 0               # 1: 圆柱视角  0: 笛卡尔视角
        self.ylabel_cnr = 'B [T]'                    # 数据单位
        self.bulk_nr = 0                             # 批处理
######################################################
        # point_Selection参数（图像不保存）
        self.point_coords = (80, 161)                # 选取点（前者是nr，默认为129）
        self.cylindrical_point = 0                   # 1: 圆柱视角  0: 笛卡尔视角
        self.ylabel_ps = 'Value'                     # Y轴（物理量）名字
######################################################
def sub_dimension_3D(config):
    # 加载数据
    try:
        with open(config.input_path, 'r') as file:
            data_str = file.read().replace('D', 'E')
        data = np.loadtxt(StringIO(data_str))
        print(f"Actual data size: {data.size}")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    try:
        if config.coordinate_system:
            expected_length = config.NX * config.NY * config.NZ
        else:
            expected_length = config.NR * config.nth * config.nz
        print(f"Expected data size: {expected_length}")
        if data.size != expected_length:
            print("Data element quantities are not as expected, check the file")
            sys.exit(1)
        print("Data element quantity checking passes")

        # 重塑数据并创建坐标网格
        if config.coordinate_system:
            array3d = data.reshape((config.NX, config.NY, config.NZ))
            # 创建笛卡尔坐标网格
            x = np.arange(config.NX)
            y = np.arange(config.NY)
            z = np.arange(config.NZ)
            x, y, z = np.meshgrid(x, y, z, indexing='ij')
        else:
            array3d = data.reshape((config.NR, config.nth, config.nz))
            # 创建圆柱坐标网格
            r = np.arange(config.NR)
            if config.semicircular:
                theta = np.linspace(0, np.pi, config.nth)  # th 从 0 到 π
            else:
                theta = np.linspace(0, 2 * np.pi, config.nth)  # th 从 0 到 2π
            z = np.arange(config.nz)
            r, theta, z = np.meshgrid(r, theta, z, indexing='ij')

            # 转换为笛卡尔坐标
            x = r * np.cos(theta)
            y = r * np.sin(theta)
    except Exception as e:
        print(f"Error while reshaping data or creating a coordinate grid: {e}")
        sys.exit(1)

    # 应用灵敏度阈值
    sensitivity_threshold = 1e-49
    mask = np.abs(array3d) >= sensitivity_threshold
    print(f"应用灵敏度阈值，保留的数据点数量: {np.sum(mask)}")

    # 根据数据确定颜色范围
    if config.data_range:
        vmin, vmax = array3d.min(), array3d.max()
        print(f"Automatic data ranges: {vmin} ~ {vmax}")
    else:
        vmin, vmax = config.vmin, config.vmax
        print(f"Customized data range: {vmin} ~ {vmax}")

    # 创建自定义颜色映射
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_data = cmap(norm(np.linspace(vmin, vmax, cmap.N)))

    # 处理零值透明
    if config.enable_zero:
        zero_index = int((0 - vmin) / (vmax - vmin) * (cmap.N - 1))
        transparent_color = np.array([1, 1, 1, 0])  # RGBA：A=0 表示透明
        if 0 <= zero_index < len(cmap_data):
            cmap_data[zero_index] = transparent_color
        print(f"已处理零值透明，zero_index={zero_index}")
    custom_cmap = mcolors.ListedColormap(cmap_data)     # 创建带有可能透明度的新颜色映射

    # 根据数据幅度设置透明度
    if config.transparency == 1:
        alpha = np.clip(np.abs(array3d) / array3d.max(), 0.02, 0.1)
    elif config.transparency == 2:
        alpha = np.clip(np.abs(array3d) / array3d.max(), 0.01, 0.05)
    elif config.transparency == 3:
        alpha = np.clip(np.abs(array3d) / array3d.max(), 0.005, 0.02)
    else:
        print("Input right transparency parameter")

    # 准备绘图数据（交换x和z轴数据）
    if config.Grid_Ratio:
        print("使用装置尺寸")
        x_plot = z[mask] * (config.physical_Z / (config.NZ - 1))
        y_plot = y[mask] * (config.physical_R / (config.NR - 1))
        z_plot = x[mask] * (config.physical_R / (config.NR - 1))
    else:
        print("使用网格尺寸")
        x_plot = z[mask]
        y_plot = y[mask]
        z_plot = x[mask]
    data_plot = array3d[mask]

    # 创建带有透明度的颜色
    colors = custom_cmap(norm(data_plot))
    colors[:, -1] = alpha[mask]

    # 创建绘图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_plot, y_plot, z_plot, color=colors, marker='o', s=1)

    # 增加空白范围的偏移量5%，防止图像越界，固定坐标轴范围，交换x和z轴数据的范围
    if config.Grid_Ratio:
        margin_x = (x_plot.max() - x_plot.min()) * 0.05
        margin_y = (y_plot.max() - y_plot.min()) * 0.05
        margin_z = (z_plot.max() - z_plot.min()) * 0.05
        ax.set_xlim([x_plot.min() - margin_x, x_plot.max() + margin_x])
        ax.set_ylim([y_plot.min() - margin_y, y_plot.max() + margin_y])
        ax.set_zlim([z_plot.min() - margin_z, z_plot.max() + margin_z])
    else:
        margin_x = (z.max() - z.min()) * 0.05
        margin_y = (y.max() - y.min()) * 0.05
        margin_z = (x.max() - x.min()) * 0.05
        ax.set_xlim([z.min() - margin_x, z.max() + margin_x])
        ax.set_ylim([y.min() - margin_y, y.max() + margin_y])
        ax.set_zlim([x.min() - margin_z, x.max() + margin_z])

    # 根据视角选项设置不同的视角
    if config.view_option == 1:
        ax.view_init(elev=0, azim=0)  # 轴向视图
    elif config.view_option == 2:
        ax.view_init(elev=0, azim=-90)  # 径向视图

    # 添加颜色条
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap)
    mappable.set_array(array3d)
    fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=5, pad=0.2)

    # 设置坐标轴标签
    ax.set_xlabel('Z')
    ax.set_ylabel('R')
    ax.set_zlabel('D')

    # 全圆半圆选择和设置轴比例
    if config.Grid_Ratio:
        ax.xaxis.set_major_locator(MultipleLocator(0.25))   # 每0.5单位一个刻度
        ax.yaxis.set_major_locator(MultipleLocator(0.193))  # 每0.193单位一个刻度
        ax.zaxis.set_major_locator(MultipleLocator(0.193))
        if config.semicircular:
            aspect_ratio = [config.physical_Z, config.physical_R, config.physical_R * 2]
        else:
            aspect_ratio = [config.physical_Z, config.physical_R * 2, config.physical_R * 2]
        ax.set_box_aspect(aspect_ratio)
    else:
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.yaxis.set_major_locator(MultipleLocator(40))
        ax.zaxis.set_major_locator(MultipleLocator(40))
        if config.semicircular:
            if config.coordinate_system:
                aspect_ratio_lattices = [config.NZ, config.NY, config.NX * 2]
            else:
                aspect_ratio_lattices = [config.nz, config.NR, config.NR * 2]
        else:
            if config.coordinate_system:
                aspect_ratio_lattices = [config.NZ, config.NY, config.NX * 2]
            else:
                aspect_ratio_lattices = [config.nz, config.NR * 2, config.NR * 2]
        ax.set_box_aspect(aspect_ratio_lattices)

    # 保存图像
    print("---------")
    random_suffix = ''.join([str(random.randint(0, 9)) for _ in range(5)])  # 生成五位随机数字的字符串
    output_png = os.path.join(config.output_path, f'output_{random_suffix}.png')    # 构建带随机数后缀的文件名
    try:
        plt.savefig(output_png, format='png', dpi=200)
        print(f"Image saved to {output_png}")
    except Exception as e:
        print(f"Error saving image: {e}")

def sub_GIF_3D(config):
    file_path = config.file_path    # 获取文件夹路径和输出路径
    fps = config.fps
    USE_GLOBAL_RANGE = config.USE_GLOBAL_RANGE
    GLOBAL_MIN = config.GLOBAL_MIN
    GLOBAL_MAX = config.GLOBAL_MAX
    file_list = sorted(glob.glob(os.path.join(file_path, '*.[tT][xX][tT]')) +
                       glob.glob(os.path.join(file_path, '*.[cC][sS][vV]')) +
                       glob.glob(os.path.join(file_path, '*.[dD][aA][tT]')) +
                       glob.glob(os.path.join(file_path, '*.[dD][aA][tT][aA]')))
    total_frames = len(file_list)
    print(f"Number of documents: {total_frames}")

    # 数据检查
    if total_frames == 0:
        print("File not found in folder")
        sys.exit(1)
    first_file = file_list[0]
    try:
        with open(first_file, 'r') as file:
            data_str = file.read().replace('D', 'E')
        data = np.loadtxt(StringIO(data_str))
        if config.coordinate_system:
            expected_length = config.NX * config.NY * config.NZ
        else:
            expected_length = config.NR * config.nth * config.nz
        print(f"Actual data size: {data.size}")
        print(f"Expected data size: {expected_length}")
        if data.size != expected_length:
            print(f"Length of the data {first_file} is not as expected, check the file")
            sys.exit(1)
        print("Data length check passes")
    except Exception as e:
        print(f"Read the first file {first_file} error: {e}")
        sys.exit(1)

    # 创建图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 动画更新函数
    def gif_update(frame_num):
        # 显示处理进度
        sys.stdout.write(f"\rprocess: {frame_num + 1}/{total_frames} ")
        sys.stdout.flush()

        ax.cla()  # 清除上一个图像
        try:
            data = np.loadtxt(file_list[frame_num])  # 读取数据
            if config.coordinate_system:
                array3d = data.reshape((config.NX, config.NY, config.NZ))  # 重塑数据为三维数组
                # 创建笛卡尔坐标网格
                x = np.arange(config.NX)
                y = np.arange(config.NY)
                z = np.arange(config.NZ)
                x, y, z = np.meshgrid(x, y, z, indexing='ij')
            else:
                array3d = data.reshape((config.NR, config.nth, config.nz))
                # 创建圆柱坐标网格
                r = np.arange(config.NR)
                if config.semicircular:
                    theta = np.linspace(0, np.pi, config.nth)  # th 从 0 到 π
                else:
                    theta = np.linspace(0, 2 * np.pi, config.nth)  # th 从 0 到 2π
                z = np.arange(config.nz)
                r, theta, z = np.meshgrid(r, theta, z, indexing='ij')

                # 转换为笛卡尔坐标
                x = r * np.cos(theta)
                y = r * np.sin(theta)

            # 过滤掉值小于阈值的部分
            sensitivity_threshold = 1e-49
            mask = np.abs(array3d) >= sensitivity_threshold

            cmap = plt.cm.jet   # 创建自定义颜色条

            # 根据开关设置颜色归一化范围
            if USE_GLOBAL_RANGE:
                norm = plt.Normalize(vmin=GLOBAL_MIN, vmax=GLOBAL_MAX)
            else:
                norm = plt.Normalize(vmin=array3d.min(), vmax=array3d.max())
            cmap_data = cmap(norm(np.linspace(norm.vmin, norm.vmax, cmap.N)))
            custom_cmap = mcolors.ListedColormap(cmap_data)     # 创建新的颜色映射

            # 根据 transparency 设置透明度
            if config.transparency_gif == 1:
                if array3d.max() == 0:
                    alpha = np.clip(np.abs(array3d) / (array3d.max() + 1e-49), 0.02, 0.04)
                    print(f"第 {frame_num + 1} 个文件最大值是 0，已添加 1e-49 偏移，否则无法设置透明度")
                else:
                    alpha = np.clip(np.abs(array3d) / array3d.max(), 0.02, 0.04)
            elif config.transparency_gif == 2:
                if array3d.max() == 0:
                    alpha = np.clip(np.abs(array3d) / (array3d.max() + 1e-49), 0.01, 0.025)
                    print(f"第 {frame_num + 1} 个文件最大值是 0，已添加 1e-49 偏移，否则无法设置透明度")
                else:
                    alpha = np.clip(np.abs(array3d) / array3d.max(), 0.01, 0.025)
            else:
                print("Input right transparency_gif parameter")

            # 数据最大为0时（说明全部为0），全值设置为0.01，避免错误
            if array3d.max() == 0:
                alpha = np.full_like(array3d, 0.01)

            # 绘制三维散点图，掩码处理后的数据，交换x和z数据
            colors = custom_cmap(norm(array3d[mask]))
            colors[:, -1] = alpha[mask]
            ax.scatter(z[mask], y[mask], x[mask], color=colors, marker='o', s=1)

            # 增加空白范围的偏移量
            margin_x = (z.max() - z.min()) * 0.05
            margin_y = (y.max() - y.min()) * 0.05
            margin_z = (x.max() - x.min()) * 0.05

            # 固定坐标轴范围，交换x和z轴数据的范围，添加偏移
            ax.set_xlim([z.min() - margin_x, z.max() + margin_x])
            ax.set_ylim([y.min() - margin_y, y.max() + margin_y])
            ax.set_zlim([x.min() - margin_z, x.max() + margin_z])

            # 设置轴比例
            if config.Grid_Ratio:
                # 装置比例
                if config.semicircular:
                    aspect_ratio = [config.physical_Z, config.physical_R, config.physical_R * 2]
                else:
                    aspect_ratio = [config.physical_Z, config.physical_R * 2, config.physical_R * 2]
                ax.set_box_aspect(aspect_ratio)
            else:
                # 网格比例
                if config.semicircular:
                    if config.coordinate_system:
                        aspect_ratio_lattices = [config.NZ, config.NY, config.NX * 2]
                    else:
                        aspect_ratio_lattices = [config.nz, config.NR, config.NR * 2]
                else:
                    if config.coordinate_system:
                        aspect_ratio_lattices = [config.NZ, config.NY, config.NX * 2]
                    else:
                        aspect_ratio_lattices = [config.nz, config.NR * 2, config.NR * 2]
                ax.set_box_aspect(aspect_ratio_lattices)

            # 设置坐标轴标签
            ax.set_xlabel('Z*2')
            ax.set_ylabel('R')
            ax.set_zlabel('D')
        except Exception as e:
            print(f"\nProcessing of file {file_list[frame_num]} error: {e}")
    ani = FuncAnimation(fig, gif_update, frames=total_frames, interval=1000 / fps)      # 创建动画

    print("------")
    print("GIF is being generated\nif interrupted, GIF based on the currently processed")
    random_suffix = ''.join([str(random.randint(0, 9)) for _ in range(5)])      # 生成五位随机数字的字符串
    output_gif = os.path.join(config.output_path, f'output_{random_suffix}.gif')      # 构建带随机数后缀的文件名
    writer = PillowWriter(fps=fps)
    try:
        ani.save(output_gif, writer=writer)
        print(f"\nGIF saved to {output_gif}")
    except Exception as e:
        print(f"\nError saving GIF: {e}")

def sub_dimension_2D(config):
    if config.output_all_files == 0:    # 单文件模式
        file_list = [config.input_path]
        os.makedirs(config.output_path, exist_ok=True)
    elif config.output_all_files == 1:  # 批量处理模式
        print("开启批量处理模式")
        output_dir = os.path.join(config.file_path, 'all_files_image')
        os.makedirs(output_dir, exist_ok=True)

        # 获取文件列表
        file_list = [os.path.join(config.file_path, file_name)
                     for file_name in os.listdir(config.file_path)
                     if os.path.isfile(os.path.join(config.file_path, file_name))]
        if not file_list:
            print(f"{config.file_path} no file")
            sys.exit(1)
        print(f"Input path: {config.file_path}\n {len(file_list)} files, process......")
        print("使用装置尺寸" if config.Grid_Ratio == 1 else "使用网格尺寸")
    else:
        print("config.output_all_files parameter error")
        sys.exit(1)

    for file_path in file_list:
        # 读取文件
        with open(file_path, 'r') as file:
            lines = file.readlines()
        ny = len(lines)
        nx = len(lines[0].split())

        # 数据处理和绘制
        if ny == 1 or nx == 1:
            if not config.output_all_files:
                print(f"识别到线性数据，长度为 {max(ny, nx)}")
            data = []
            for line in lines:
                values = line.split()
                data.extend([float(value.replace('D', 'E')) for value in values])
            data = np.array(data)
            if config.log_scale:
                print("开启log运算")
                data = np.log(np.where(data <= 0, 1e-49, data))
            plt.xlabel(config.xlabel)
            plt.ylabel(config.ylabel)
            plt.plot(data)
        elif nx == 2:
            if not config.output_all_files:
                print(f"识别到双层线性数据，长度为 {ny} ，多层折线图功能不支持log运算")
            data = []
            for line in lines:
                values = line.split()
                data.append([float(value.replace('D', 'E')) for value in values])
            data = np.array(data)
            plt.xlabel(config.xlabel)
            plt.ylabel(config.ylabel)
            plt.plot(data[:, 0], data[:, 1])
            # 添加虚线参考线标记
            #cutoff_x1 = 0.1169  # x 位置
            #plt.axvline(x=cutoff_x1, color='b', linestyle='--', linewidth=1.5, label=f'R={cutoff_x1}[m]')
            #cutoff_x2 = 0.091  # x 位置
            #plt.axvline(x=cutoff_x2, color='b', linestyle='--', linewidth=1.5, label=f'R={cutoff_x2}[m]')
            #区域染色
            #plt.axvspan(cutoff_x1, cutoff_x2, color='gray', alpha=0.2, label="Wave penetration")
            #cutoff_y = 0.88    # y 位置
            #plt.axhline(y=cutoff_y, color='r', linestyle='--', linewidth=1.5, label=f'β={cutoff_y}')
            plt.legend()
        elif nx == 3:
            if not config.output_all_files:
                print(f"识别到三层线性数据，长度为 {ny} ,多层折线图功能不支持log运算")
            data = []
            for line in lines:
                values = line.split()
                data.append([float(value.replace('D', 'E')) for value in values])
            data = np.array(data)
            plt.xlabel(config.xlabel)
            plt.ylabel(config.ylabel)
            plt.plot(data[:, 0], data[:, 1], label=config.Line_1, linestyle='--', marker='o')  # 第一条折线
            plt.plot(data[:, 0], data[:, 2], label=config.Line_2, linestyle='-', marker='x')  # 第二条折线
            plt.legend()  # 显示图例
        else:
            if not config.output_all_files:
                print(f"识别到二维平面数据，尺寸为 ({nx}, {ny})")
                print("使用装置尺寸" if config.Grid_Ratio == 1 else "使用网格尺寸")
            if config.cylindrical == 0:
                data = np.zeros((ny, nx))
                for i, line in enumerate(lines):
                    if not line.strip():
                        continue
                    values = line.split()
                    for j in range(nx):
                        data[i, j] = float(values[j].replace('D', 'E'))
                if config.Grid_Ratio == 1:
                    extent = [0, config.physical_Z, 0, config.physical_R]
                    plt.xlabel(f"{config.xlabel} ({config.physical_Z})")
                    plt.ylabel(f"{config.ylabel} ({config.physical_R})")
                else:
                    extent = [0, nx, 0, ny]
                    plt.xlabel(config.xlabel)
                    plt.ylabel(config.ylabel)
                plt.imshow(data, cmap='jet', origin='lower', extent=extent, aspect='equal')

                if config.Grid_Ratio == 1:
                    if config.physical_R == 0.3857:
                        cbar = plt.colorbar(label=config.colorbar_label, shrink=0.3)
                        cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 3))
                    else:
                        cbar = plt.colorbar(label=config.colorbar_label, shrink=0.35)
                        cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 5))
                else:
                    cbar = plt.colorbar(label=config.colorbar_label, shrink=0.5)
                    cbar.set_ticks(np.linspace(cbar.vmin, cbar.vmax, 5))
            elif config.cylindrical == 1:
                print("识别到圆柱坐标系数据，正在处理...")
                nr = ny
                nth = nx
                data = np.zeros((nr, nth))
                # 读取数据文件
                for i, line in enumerate(lines):
                    if not line.strip():
                        continue
                    values = line.split()
                    for j in range(nth):
                        data[i, j] = float(values[j].replace('D', 'E'))

                # 创建极坐标网格
                max_radius = nr
                r = np.linspace(0, max_radius, nr)  # 半径范围
                theta = np.linspace(0, 2 * np.pi, nth)  # 角度范围
                R, Theta = np.meshgrid(r, theta, indexing='ij')

                # 转换为笛卡尔坐标
                X = R * np.cos(Theta)
                Y = R * np.sin(Theta)

                # 将极坐标数据映射到笛卡尔网格
                x_cart = np.linspace(-max_radius, max_radius, 500)
                y_cart = np.linspace(-max_radius, max_radius, 500)
                X_cart, Y_cart = np.meshgrid(x_cart, y_cart)

                # 插值极坐标数据到笛卡尔网格
                points = np.column_stack((X.ravel(), Y.ravel()))
                values = data.ravel()
                Z_cart = griddata(points, values, (X_cart, Y_cart), method='linear', fill_value=0)
                Z_cart_rot = np.rot90(Z_cart, k=-1)     # 0度角默认三点钟方向，逆时针旋转90度

                if config.Grid_Ratio == 1:
                    extent = [-config.physical_R, config.physical_R, -config.physical_R, config.physical_R]
                    plt.xlabel(f"{config.xlabel} ({config.physical_R})")
                    plt.ylabel(f"{config.ylabel} ({config.physical_R})")
                else:
                    plt.xlabel(config.ylabel)
                    plt.ylabel(config.ylabel)
                    extent = [-max_radius, max_radius, -max_radius, max_radius]

                # 创建掩膜，将圆外部设置为白色
                if config.white_mask:
                    print("创建装置外部掩膜")
                    mask = np.sqrt(X_cart ** 2 + Y_cart ** 2) > max_radius
                    Z_cart_rot[mask] = np.nan       # 圆外部设置为空值，默认显示为白色

                # 显示图像
                plt.imshow(Z_cart_rot, cmap='jet', origin='lower', extent=extent, aspect='equal')

                # 添加圆形辅助标记
                circle_radius = max_radius if config.Grid_Ratio != 1 else config.physical_R
                circle = plt.Circle((0, 0), circle_radius, color='black', fill=False, linestyle='-', linewidth=1.5)
                plt.gca().add_artist(circle)  # 在当前坐标轴添加圆形

                # 确保坐标轴范围匹配圆形
                plt.gca().set_xlim(-circle_radius, circle_radius)
                plt.gca().set_ylim(-circle_radius, circle_radius)

                # 颜色条标注出问题，将下面替换为 plt.colorbar(label=config.colorbar_label, pad=0.1
                cb = plt.colorbar(label=config.colorbar_label, pad=0.1, format=ticker.ScalarFormatter())
                cb.formatter.set_scientific(True)
                cb.formatter.set_powerlimits((-2, 2))
                cb.update_ticks()
            else:
                print("Error cylindrical parameters")
        plt.title(config.title)

        # 保存或显示图像
        if config.output_all_files == 0:
            if config.File_save:
                try:
                    random_suffix = ''.join([str(random.randint(0, 9)) for _ in range(5)])      # 随机生成五位数字

                    # 提取原文件名（不带扩展名），生成新文件名，添加随机后缀和新扩展名
                    base_name = os.path.basename(file_path).rsplit('.', 1)[0]
                    output_path = os.path.join(config.output_path, f"{base_name}_{random_suffix}.png")
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    print(f"Image has been saved to：{output_path}")
                except Exception as e:
                    print(f"Error saving image: {e}")
            else:
                print("Image saving is not enabled")
            plt.show()
        elif config.output_all_files == 1:
            output_path = os.path.join(output_dir, os.path.basename(file_path).rsplit('.', 1)[0] + '.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            print("Error output_all_files parameters")
        plt.close()
    if config.output_all_files:
        print(f"All image has been saved to:{output_dir}")

def sub_GIF_2D(config):
    # 获取文件列表并按名称排序
    file_list = sorted([os.path.join(config.file_path, file_name)
                        for file_name in os.listdir(config.file_path)])
    if not file_list:
        print(f"Path {config.file_path} no files")
        sys.exit(1)
    print("输入圆柱坐标系" if config.cylindrical_g else "输入笛卡尔坐标系")
    print("使用装置尺寸" if config.Grid_Ratio else "使用网格尺寸")
    print("使用全局范围" if config.use_manual_range else "使用动态范围")
    print("---------")

    # 创建GIF动画
    fig, ax = plt.subplots(figsize=(6, 6))
    writer = PillowWriter(fps=config.FPS)
    gif_path = os.path.join(config.output_path, "output.gif")
    writer.setup(fig, gif_path, dpi=150)

    for idx, file_path in enumerate(file_list, start=1):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        ny = len(lines)
        nx = len(lines[0].split())

        # 读取并解析数据
        data = np.zeros((ny, nx))
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            values = line.split()
            for j in range(nx):
                data[i, j] = float(values[j].replace('D', 'E'))

        # 对数处理
        if config.log_scale_g:
            data = np.log(np.where(data <= 0, 1e-49, data))

        # 显示处理进度
        sys.stdout.write(f"\rProcess: {idx}/{len(file_list)} : {os.path.basename(file_path)}")
        sys.stdout.flush()

        # 动态或手动设置颜色条范围
        if config.use_manual_range:
            vmin, vmax = config.manual_min, config.manual_max
        else:
            vmin, vmax = data.min(), data.max()

        ax.clear()
        if config.cylindrical_g == 0:
            extent = [0, data.shape[1], 0, data.shape[0]] if config.Grid_Ratio == 0 else [0, config.physical_Z, 0,
                                                                                          config.physical_R]
            im = ax.imshow(data, cmap='jet', origin='lower', extent=extent, aspect='equal', vmin=vmin, vmax=vmax)
            ax.set_xlabel(config.xlabel_g)
            ax.set_ylabel(config.ylabel_g)
        elif config.cylindrical_g == 1:
            nr, nth = data.shape
            r = np.linspace(0, nr, nr)
            theta = np.linspace(0, 2 * np.pi, nth)
            R, Theta = np.meshgrid(r, theta, indexing='ij')
            X = R * np.cos(Theta)
            Y = R * np.sin(Theta)

            # 插值到笛卡尔网络
            x_cart = np.linspace(-nr, nr, 500)
            y_cart = np.linspace(-nr, nr, 500)
            X_cart, Y_cart = np.meshgrid(x_cart, y_cart)
            points = np.column_stack((X.ravel(), Y.ravel()))
            values = data.ravel()
            Z_cart = griddata(points, values, (X_cart, Y_cart), method='linear', fill_value=0)
            Z_cart_rot = np.rot90(Z_cart, k=-1)  # 0度角默认三点钟方向，逆时针旋转90度
            extent = [-nr, nr, -nr, nr] if config.Grid_Ratio == 0 else [-config.physical_R, config.physical_R,
                                                                        -config.physical_R, config.physical_R]
            im = ax.imshow(Z_cart_rot, cmap='jet', origin='lower', extent=extent, aspect='equal', vmin=vmin, vmax=vmax)
            ax.set_xlabel(config.ylabel_g)
            ax.set_ylabel(config.ylabel_g)
        else:
            print("input right cylindrical_g parameter")
        cbar = fig.colorbar(im, ax=ax, label=config.colorbar_label_g, shrink=0.25)
        cbar.ax.set_ylabel(config.colorbar_label_g)
        plt.title(os.path.basename(file_path))
        writer.grab_frame()
        cbar.remove()       # 移除颜色条，避免叠加
    print("GIF file being saved...")
    writer.finish()
    print(f"Data shape: {data.shape}")
    print(f"GIF file saved to：{gif_path}")

def sub_single_nz_nth(config):
    # 批处理模式
    if config.bulk_nz_nth:
        out_dir = os.path.join(config.file_path, "bulk_nz_nth_results")     # 创建结果文件夹
        os.makedirs(out_dir, exist_ok=True)

        # 遍历该目录下的所有文件，比如根据后缀判断读取的文件
        for filename in os.listdir(config.file_path):
            file_full_path = os.path.join(config.file_path, filename)
            if os.path.isfile(file_full_path):
                try:
                    try:
                        with open(file_full_path, 'r') as file:
                            data_str = file.read().replace('D', 'E')
                        data = np.loadtxt(StringIO(data_str))
                        if config.cylindrical_singel:
                            print("输入为圆柱数据")
                            array2D = data.reshape(config.nr_up, config.nth_up)
                            if data.size != config.nr_up * config.nth_up:
                                print(f"数据长度与二维模式预期不符: {file_full_path}")
                                sys.exit(1)
                        else:
                            print("输入为笛卡尔数据")
                            array2D = data.reshape(config.NR, config.nz)
                            if data.size != config.NR * config.nz:
                                print(f"数据长度与二维模式预期不符: {file_full_path}")
                                sys.exit(1)
                    except Exception as e:
                        print(f"Error loading data: {e}")
                        continue        # 不终止整个程序，继续处理下一个文件
                    adjusted_nz = config.choice_nz_nth - 1
                    if config.cylindrical_singel:
                        if adjusted_nz < 0 or adjusted_nz >= config.nth_up:
                            print(f"choice_nz_nth 超出范围 (1, {config.nth_up})")
                            sys.exit(1)
                    else:
                        if adjusted_nz < 0 or adjusted_nz >= config.nz:
                            print(f"choice_nz_nth 超出范围 (1, {config.nz})")
                            sys.exit(1)
                    selected_column = array2D[:, adjusted_nz]
                    num_rows, num_cols = array2D.shape
                    print(f"{filename} | Array Shape: ({num_rows}, {num_cols})\nSelection Index:  {config.choice_nz_nth}")
                    max_value = np.max(selected_column)
                    min_value = np.min(selected_column)
                    plt.figure(figsize=(16, 4))     # 图像比例
                    ax = plt.gca()
                    ax.yaxis.set_major_locator(LinearLocator(numticks=6))  # 显示6个主刻度
                    if config.Grid_Ratio == 1:
                        x_physical = np.linspace(0, config.physical_R, len(selected_column))
                        plt.plot(x_physical, selected_column, marker='o', linestyle='-', markersize=4)
                    else:
                        plt.plot(range(config.nr_up), selected_column, marker='o', linestyle='-', markersize=4)
                    if config.title_clear == 0:
                        if config.cylindrical_singel:
                            plt.title(f'Line Plot for nth={config.choice_nz_nth}(grid)')
                        else:
                            plt.title(f'Line Plot for nz={config.choice_nz_nth}(grid)')
                    if num_rows == 257:
                        plt.xlabel('r[m]*2(0.7714)')
                    elif num_rows == 129:
                        plt.xlabel('r[m](0.3857)')
                    else:
                        plt.xlabel('r[m]')
                    plt.ylabel(config.ylabel_cnznth)
                    plt.grid(True)
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                    # 图像中显示最大值和最小值
                    plt.text(0.95, 0.95, f'Max: {max_value:.16f}', transform=plt.gca().transAxes,
                             fontsize=10, verticalalignment='top', horizontalalignment='right', color='red')
                    plt.text(0.95, 0.90, f'Min: {min_value:.16f}', transform=plt.gca().transAxes,
                             fontsize=10, verticalalignment='top', horizontalalignment='right', color='blue')
                    save_name = os.path.splitext(filename)[0] + "_nz_nth.png"
                    plt.savefig(os.path.join(out_dir, save_name), dpi=150)
                    plt.close()
                except Exception as e:
                    print(f"Error drawing a single nz : {file_full_path}）: {e}")
                    continue
    else:
        try:
            try:
                with open(config.input_path, 'r') as file:
                    data_str = file.read().replace('D', 'E')
                data = np.loadtxt(StringIO(data_str))
                if config.cylindrical_singel:
                    print("输入为圆柱数据")
                    array2D = data.reshape(config.nr_up, config.nth_up)
                    if data.size != config.nr_up * config.nth_up:
                        print(f"数据长度与二维模式预期不符: {config.input_path}")
                        sys.exit(1)
                else:
                    print("输入为笛卡尔数据")
                    array2D = data.reshape(config.NR, config.nz)
                    if data.size != config.NR * config.nz:
                        print(f"数据长度与二维模式预期不符: {config.input_path}")
                        sys.exit(1)
            except Exception as e:
                print(f"Error loading data: {e}")
                sys.exit(1)
            adjusted_nz = config.choice_nz_nth - 1
            if config.cylindrical_singel:
                if adjusted_nz < 0 or adjusted_nz >= config.nth_up:
                    print(f"choice_nz_nth 超出范围 (1, {config.nth_up})")
                    sys.exit(1)
            else:
                if adjusted_nz < 0 or adjusted_nz >= config.nz:
                    print(f"choice_nz_nth 超出范围 (1, {config.nz})")
                    sys.exit(1)
            selected_column = array2D[:, adjusted_nz]
            num_rows, num_cols = array2D.shape
            print(f"Array shape: ({num_rows}, {num_cols})")
            print(f"Selected location: nz = {config.choice_nz_nth}")
            max_value = np.max(selected_column)
            min_value = np.min(selected_column)
            plt.figure(figsize=(16, 4))
            ax = plt.gca()
            ax.yaxis.set_major_locator(LinearLocator(numticks=6)) # 显示6个主刻度
            if config.Grid_Ratio == 1:
                x_physical = np.linspace(0, config.physical_R, len(selected_column))
                plt.plot(x_physical, selected_column, marker='o', linestyle='-', markersize=4)
            else:
                plt.plot(range(config.nr_up), selected_column, marker='o', linestyle='-', markersize=4)
            if config.title_clear == 0:
                if config.cylindrical_singel:
                    plt.title(f'Line Plot for nth={config.choice_nz_nth}(grid)')
                else:
                    plt.title(f'Line Plot for nz={config.choice_nz_nth}(grid)')
            if num_rows == 257:
                plt.xlabel('r[m]*2(0.7714)')
            elif num_rows == 129:
                plt.xlabel('r[m](0.3857)')
            else:
                plt.xlabel('r[m]')
            plt.ylabel(config.ylabel_cnznth)
            plt.grid(True)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            # 添加虚线参考线标记波形截止区
            #cutoff_x = 0.01  # 截止区 x 位置
            #plt.axvline(x=cutoff_x, color='r', linestyle='--', linewidth=3, label='Separatrix')
            #cutoff_y = 0.88  # 截止区 y 位置
            #plt.axhline(y=cutoff_y, color='r', linestyle='--', linewidth=3, label='Y-Cutoff')
            #plt.legend()

            # 图像中显示最大值和最小值
            plt.text(0.95, 0.95, f'Max: {max_value:.16f}', transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top', horizontalalignment='right', color='red')
            plt.text(0.95, 0.90, f'Min: {min_value:.16f}', transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top', horizontalalignment='right', color='blue')
            plt.show()
        except Exception as e:
            print(f"Error drawing a single nz: {e}")
            sys.exit(1)

def sub_single_nr(config):
    # 批处理模式
    if config.bulk_nr:
        out_dir = os.path.join(config.file_path, "bulk_nr_results")
        os.makedirs(out_dir, exist_ok=True)
        for filename in os.listdir(config.file_path):
            file_full_path = os.path.join(config.file_path, filename)
            if os.path.isfile(file_full_path):
                try:
                    try:
                        with open(file_full_path, 'r') as file:
                            data_str = file.read().replace('D', 'E')
                        data = np.loadtxt(StringIO(data_str))
                        if config.cylindrical_singel_nr:
                            print("输入圆柱数据")
                            array2D = data.reshape(config.nr_up, config.nth_up)
                            if data.size != config.nr_up * config.nth_up:
                                print(f"数据长度与二维模式预期不符: {file_full_path}")
                                continue
                        else:
                            print("输入笛卡尔数据")
                            array2D = data.reshape(config.NR, config.nz)
                            if data.size != config.NR * config.nz:
                                print(f"数据长度与二维模式预期不符: {file_full_path}")
                                continue
                    except Exception as e:
                        print(f"Error loading data： {e}")
                        continue
                    adjust_nr = config.choice_nr - 1
                    if config.cylindrical_singel_nr:
                        if adjust_nr < 0 or adjust_nr >= config.nr_up:
                            print(f"choice_nr 超出范围 (1, {config.nr_up})")
                            sys.exit(1)
                    else:
                        if adjust_nr < 0 or adjust_nr >= config.NR:
                            print(f"choice_nr 超出范围 (1, {config.NR})")
                            sys.exit(1)
                    selected_column = array2D[adjust_nr, :]
                    num_rows, num_cols = array2D.shape
                    print(f"{filename} | Array shape: ({num_rows}, {num_cols})\nSelection Index: {config.choice_nr}")
                    max_value = np.max(selected_column)
                    min_value = np.min(selected_column)
                    max_index = np.argmax(selected_column)
                    min_index = np.argmin(selected_column)
                    if config.cylindrical_singel_nr:
                        print(f"Maximum position: nth = {max_index + 1}\nMinimum position: nth = {min_index + 1}")
                    else:
                        max_position = max_index * config.physical_Z / (len(selected_column) - 1)
                        min_position = min_index * config.physical_Z / (len(selected_column) - 1)
                        print(f"Maximum position: z = {max_position:.16f} [m]\nMinimum position: z = {min_position:.16f} [m]")
                    plt.figure(figsize=(16, 4))
                    ax = plt.gca()
                    ax.yaxis.set_major_locator(LinearLocator(numticks=6))  # 显示6个主刻度
                    if config.cylindrical_singel_nr:
                        plt.xlabel("θ numbers")
                        x_grid = range(len(selected_column))
                        plt.plot(x_grid, selected_column, marker='o', linestyle='-', markersize=4)
                    else:
                        plt.xlabel("z[m](2.24)")
                        x_physical = np.linspace(0, config.physical_Z, len(selected_column))
                        plt.plot(x_physical, selected_column, marker='o', linestyle='-', markersize=4)
                    if config.title_clear == 0:
                        plt.title(f'Line plot for nr={config.choice_nr}(grid)')
                    plt.ylabel(config.ylabel_cnr)
                    plt.grid(True)
                    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                    plt.text(0.95, 0.95, f'Max: {max_value:.16f}', transform=plt.gca().transAxes,
                             fontsize=10, verticalalignment='top', horizontalalignment='right', color='red')
                    plt.text(0.95, 0.90, f'Min: {min_value:.16f}', transform=plt.gca().transAxes,
                             fontsize=10, verticalalignment='top', horizontalalignment='right', color='blue')
                    save_name = os.path.splitext(filename)[0] + "_nr.png"
                    plt.savefig(os.path.join(out_dir, save_name), dpi=150)
                    plt.close()
                except Exception as e:
                    print(f"Error drawing a single nr : {file_full_path}）: {e}")
                    continue
    else:
        try:
            try:
                with open(config.input_path, 'r') as file:
                    data_str = file.read().replace('D', 'E')
                data = np.loadtxt(StringIO(data_str))
                if config.cylindrical_singel_nr:
                    print("输入圆柱数据")
                    array2D = data.reshape(config.nr_up, config.nth_up)
                    if data.size != config.nr_up * config.nth_up:
                        print(f"数据长度与二维模式预期不符: {config.input_path}")
                        sys.exit(1)
                else:
                    print("输入笛卡尔数据")
                    array2D = data.reshape(config.NR, config.nz)
                    if data.size != config.NR * config.nz:
                        print(f"数据长度与二维模式预期不符: {config.input_path}")
                        sys.exit(1)
            except Exception as e:
                print(f"Error loading data： {e}")
                sys.exit(1)
            adjust_nr = config.choice_nr - 1
            if config.cylindrical_singel_nr:
                if adjust_nr < 0 or adjust_nr >= config.nr_up:
                    print(f"choice_nr 超出范围 (1， {config.nr_up})")
                    sys.exit(1)
            else:
                if adjust_nr < 0 or adjust_nr >= config.NR:
                    print(f"choice_nr 超出范围 (1， {config.NR})")
                    sys.exit(1)
            selected_column = array2D[adjust_nr, :]
            num_rows, num_cols = array2D.shape
            print(f"Array shape: ({num_rows}, {num_cols})")
            print(f"Selected position: nr = {config.choice_nr}")
            max_value = np.max(selected_column)
            min_value = np.min(selected_column)

            # 打印最大值和最小值的位置和对应值
            max_index = np.argmax(selected_column)
            min_index = np.argmin(selected_column)
            if config.cylindrical_singel_nr:
                print(f"Maximum position: nth = {max_index + 1}")
                print(f"Minimum position: nth = {min_index + 1}")
            else:
                max_position = max_index * config.physical_Z / (len(selected_column) - 1)
                min_position = min_index * config.physical_Z / (len(selected_column) - 1)
                print(f"Maximum position: z = {max_position:.16f} [m]")
                print(f"Minimum position: z = {min_position:.16f} [m]")
            plt.figure(figsize=(16, 4))
            ax = plt.gca()
            ax.yaxis.set_major_locator(LinearLocator(numticks=6))  # 显示6个主刻度
            if config.cylindrical_singel_nr:
                plt.xlabel("θ numbers")
                x_grid = range(len(selected_column))
                plt.plot(x_grid, selected_column, marker='o', linestyle='-', markersize=4)
            else:
                plt.xlabel("z[m](2.24)")
                x_physical = np.linspace(0, config.physical_Z, len(selected_column))
                plt.plot(x_physical, selected_column, marker='o', linestyle='-', markersize=4)
            if config.title_clear == 0:
                plt.title(f'Line plot for nr={config.choice_nr}(grid)')
            plt.ylabel(config.ylabel_cnr)
            plt.grid(True)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            # 添加虚线参考线标记波形截止区
            # cutoff_x = 0.49  # 实际的截止区 x 位置
            # plt.axvline(x=cutoff_x, color='r', linestyle='--', linewidth=3, label='Separatrix')
            # plt.legend()

            # 图像中显示最大值和最小值
            plt.text(0.95, 0.95, f'Max: {max_value:.16f}', transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top', horizontalalignment='right', color='red')
            plt.text(0.95, 0.90, f'Min: {min_value:.16f}', transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='top', horizontalalignment='right', color='blue')
            plt.show()
        except Exception as e:
            print(f"Error drawing a single nr: {e}")
            sys.exit(1)

def sub_point_Selection(config):
    try:
        point_nr, point_nz = config.point_coords
        point_values = []
        file_list = sorted(os.listdir(config.file_path), key=lambda x: int(os.path.splitext(x)[0]))
        total_files = len(file_list)
        for idx, file_name in enumerate(file_list, 1):
            sys.stdout.write(f"\rProcess {idx}/{total_files} : {file_name}")
            sys.stdout.flush()
            try:
                file_path = os.path.join(config.file_path, file_name)
                with open(file_path, 'r') as file:
                    data_str = file.read().replace('D', 'E')
                data = np.loadtxt(StringIO(data_str))
                if config.cylindrical_point:
                    if data.size != config.NR * config.nth_up:
                        print(f"数据长度与二维模式预期不符: {file_path}")
                        sys.exit(1)
                    array2D = data.reshape(config.NR, config.nth_up)
                else:
                    if data.size != config.NR * config.nz:
                        print(f"数据长度与二维模式预期不符: {file_path}")
                        sys.exit(1)
                    array2D = data.reshape(config.NR, config.nz)
            except Exception as e:
                print(f"Error loading data: {e}")
                sys.exit(1)
            value = array2D[point_nr - 1, point_nz - 1]  # 获取指定点的值
            point_values.append(value)
        x_axis = range(1, len(point_values) + 1)
        plt.figure()
        plt.plot(x_axis, point_values, marker='o', linestyle='-', markersize=4)
        plt.xlabel('Time steps')
        plt.ylabel(f"{config.ylabel_ps}")
        plt.title(f'Point Value at ({point_nr}, {point_nz}) across Files')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Error when plotting a line graph with specified points: {e}")
        sys.exit(1)


def main():
    config = Config()
    print("###############################")
    variables = [
        # 功能封装列表
        config.dimension_2D, config.dimension_3D, config.GIF_2D, config.GIF_3D, 
        config.single_nz_nth, config.single_nr, config.point_Selection
    ]
    ones_count = sum(variables)
    if any(v not in (0, 1) for v in variables):
        print("Please select the correct function")
        sys.exit(1)
    elif ones_count > 1:
        print("Prevent path conflicts, resulting in data loss, it is not recommended to turn on more than one function")
        sys.exit(1)
    elif ones_count == 0:
        print("Please select a function")
        sys.exit(1)
    else:
        if config.dimension_2D:
            print("2D drawing function:")
            sub_dimension_2D(config)
        if config.dimension_3D:
            print("3D drawing function:")
            sub_dimension_3D(config)
        if config.GIF_2D:
            print("2D GIF Generation Function:")
            sub_GIF_2D(config)
        if config.GIF_3D:
            print("3D GIF Generation Function:")
            sub_GIF_3D(config)
        if config.single_nz_nth:
            print("Displays a selection of nz or nth data line graph functions:")
            sub_single_nz_nth(config)
        if config.single_nr:
            print("Displays a selection of nr data line graph functions:")
            sub_single_nr(config)
        if config.point_Selection:
            print("Data evolution over time function for selected points:")
            sub_point_Selection(config)

if __name__ == "__main__":
    main()
