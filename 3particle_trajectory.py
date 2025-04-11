from HEAD import *

class Config:
    def __init__(self):
######################################################
        # paradigm 参数说明：
        # 0 - 显示单个粒子轨道（静态图）
        # 1 - 显示多个粒子散点（束流静态图）
        # 2 - 显示单个粒子轨道（动态图）
        # 3 - 显示多个粒子散点（束流动态图）
        self.paradigm = 3
        self.x_columns = 1              # X坐标数据位于第几列（如填写1，则Y在第二列，Z在第三列）
        self.fps = 15                   # 动态图帧率
        self.plt_dpi = 200              # gif清晰度
        self.paint_step = 10            # 数据步长：由于数据暂存量太大，会导致爆内存，当单个粒子轨道图时，实际为动态图采样步长，
                                        # 而束流动态图时，实际为文件读取步长，3400个文件情况下，电脑32G内存最多设置为10，等比类推
        
        # 文件路径
        self.particle_filename = r"D:\ZHANGSIYU\CARA\单粒子轨道结果.txt"                  # 粒子数据（单粒子单文件）
        self.particle_folder = r"D:\ZHANGSIYU\M1后_离子空间电荷扩散CARA\结果\1000质子\output"            # 粒子数据（多粒子文件夹）
        self.output_path = r"C:\Users\Administrator\Desktop"                            # 输出路径, 默认桌面

        # 背景场
        self.add_field = 0          # 加载场数据
        self.field_filename = r"D:\ZHANGSIYU\CARA离子空间电荷扩散\教授结果\MICROAVS.txt"    # 背景场数据
        self.field_alpha = 0.1          # 背景场透明度
        self.field_shape = (129, 257)   # 背景场数组

        # 束斑标靶
        self.spot_open = 1      # 1 开启
        self.spot_R = 0.08      # 束斑标靶参考半径
        self.spot_file = r"D:\ZHANGSIYU\M1后_离子空间电荷扩散CARA\结果\1000质子\spot.txt"   # 束斑标靶spot文件路径
        
######################################################
def load_data(filename, x_columns):
    try:
        data = np.loadtxt(filename)
        if data.shape[1] < 3:
            raise ValueError("（x y z）数据位于1 2 3列")
        x, y, z = data[:, x_columns - 1], data[:, x_columns], data[:, x_columns + 1]
        return x, y, z
    except ValueError as ve:
        print(f"Data reading error: {ve}")
        sys.exit(1)

def load_field(filename, xx, yy):
    try:
        data = np.loadtxt(filename)
        field_values = data[:, 0]
        if field_values.size != xx * yy:
            xyxy = xx * yy
            raise ValueError(f"Field data row mismatch: expected {xyxy} ，Actual {field_values.size} row")
        field_array = field_values.reshape((xx, yy))
        print("Field data loaded successfully")
        return field_array
    except ValueError as ve:
        print(f"Data reading error: {ve}")
        sys.exit(1)

def plot_spot_distribution(config):
    try:
        data = np.loadtxt(config.spot_file)
        if data.shape[1] != 2:
            raise ValueError("束斑数据需要严格包含两列（x y）")
        x_all, y_all = data[:, 0], data[:, 1]
    except Exception as e:
        print(f"Data loading error: {e}")
        sys.exit(1)
    
    # 计算粒子距离
    r_all = np.sqrt(x_all**2 + y_all**2)
    n_inside = np.sum(r_all <= config.spot_R)
    n_outside = np.sum(r_all > config.spot_R)

    # 绘图+粒子+参考边界
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x_all, y_all, c='blue', s=5, label='Particles')
    circle = plt.Circle((0, 0), config.spot_R, fill=False, color='gray', linewidth=0.5, 
                       label=f'Reference R={config.spot_R:.2f}')
    ax.add_patch(circle)
    ax.set_xlim(-0.1, 0.1)          # 设置坐标轴范围
    ax.set_ylim(-0.1, 0.1)
    ax.set_aspect('equal')          # 保持圆形不变形
    ax.set_xticks([-0.09, -0.06, -0.03, 0, 0.03, 0.06, 0.09])   # 手动设置刻度
    ax.set_yticks([-0.09, -0.06, -0.03, 0, 0.03, 0.06, 0.09])
    ax.legend(loc='upper right')    # 图例
    print(f"Statistical results (R={config.spot_R}):")
    print(f"Number of inside particles: {n_inside}")
    print(f"Number of outside particles: {n_outside}")
    plt.show()

def plot_particles(x, y, z, paradigm, R=0.12, z_min=0, z_max=1.7, field_data=None, field_alpha=0.5):
    # R=0.12 不是装置实际半径，而是方便观察轨道而设置的，防止电磁波会被截止，实际半径为 2m
    # 因为装置右侧是Z方向，与默认matplotlib不一致，所以进行重新映射
    new_x = z
    new_y = x
    new_z = y

    fig = plt.figure(figsize=(10, 6))               # 窗口大小
    ax = fig.add_subplot(111, projection='3d')      # 创建3D图形

    # 绘制背景场
    if field_data is not None:
        # 构造网格：
        x_grid = np.linspace(z_min, z_max, field_data.shape[1])     # new_x 轴（原 z 轴）：从 z_min 到 z_max
        z_grid = np.linspace(-R, R, field_data.shape[0])            # new_z 轴（原 x 轴）：从 -R 到 R
        X_mesh, Z_mesh = np.meshgrid(x_grid, z_grid)
        Y_mesh = np.zeros_like(X_mesh)
        cmap = plt.get_cmap('jet')
        norm = Normalize(vmin=field_data.min(), vmax=field_data.max())
        facecolors = cmap(norm(field_data))         # 生成颜色数组，调整 alpha 通道
        facecolors[..., 3] = field_alpha            # 透明度设置不生效，强制设置透明度（可能是库版本问题）
        surf = ax.plot_surface(X_mesh, Y_mesh, Z_mesh, rstride=1, cstride=1, facecolors=facecolors, shade=True)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(field_data)
        ticks = np.linspace(field_data.min(), field_data.max(), 10)      # 颜色条刻度数量
        fig.colorbar(mappable, ax=ax, shrink=0.2, aspect=10, pad=0.1, label='PSI', ticks=ticks)

    # 绘制粒子数据
    if paradigm == 1:
        ax.scatter(new_x, new_y, new_z, c='blue', marker='o', s=20, label='Particles')          # 绘制粒子位置
    else:
        ax.plot(new_x, new_y, new_z, label='particle trajectory', color='blue', lw=2)     # 绘制例子轨迹，lw轨迹线宽度

    ax.set_xlabel('Z', labelpad=100)
    ax.set_ylabel('X', labelpad=10)
    ax.set_zlabel('Y', labelpad=10)
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(-R, R)
    ax.set_zlim(-R, R)
    ax.set_box_aspect([z_max - z_min, 2 * R, 2 * R])
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_yticks([-0.12, -0.06, 0, 0.06, 0.12])            # 手动设置刻度
    ax.set_zticks([-0.12, -0.06, 0, 0.06, 0.12])

    # 绘制圆柱外壳
    theta = np.linspace(0, 2 * np.pi, 50)
    new_x_vals = np.linspace(z_min, z_max, 50)
    theta_grid, new_x_grid = np.meshgrid(theta, new_x_vals)
    new_y_cyl = R * np.cos(theta_grid)
    new_z_cyl = R * np.sin(theta_grid)
    ax.plot_wireframe(new_x_grid, new_y_cyl, new_z_cyl, color='gray', alpha=0.1)    # alpha 装置外壳透明度
    ax.legend(bbox_to_anchor=(1, 0.7))
    plt.tight_layout()
    plt.show()

def generate_trajectory_gif(particle_filename, output_folder, field_data=None, field_alpha=0.5, 
                            R=0.12, z_min=0, z_max=1.7, x_columns=1, fps=10, paint_step=10):
    x, y, z = load_data(particle_filename, x_columns)      # 从单个文件中加载所有点（每一行一个点）
    new_x = z
    new_y = y
    new_z = x
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制背景场
    if field_data is not None:
        x_grid = np.linspace(z_min, z_max, field_data.shape[1])
        z_grid = np.linspace(-R, R, field_data.shape[0])
        X_mesh, Z_mesh = np.meshgrid(x_grid, z_grid)
        Y_mesh = np.zeros_like(X_mesh)
        cmap = plt.get_cmap('jet')
        norm = Normalize(vmin=field_data.min(), vmax=field_data.max())
        facecolors = cmap(norm(field_data))
        facecolors[..., 3] = field_alpha
        ax.plot_surface(X_mesh, Y_mesh, Z_mesh, rstride=1, cstride=1, facecolors=facecolors, shade=True)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        #mappable.set_array(field_data)
        mappable.set_array([])
        ticks = np.linspace(field_data.min(), field_data.max(), 10)
        fig.colorbar(mappable, ax=ax, shrink=0.2, aspect=10, pad=0.1, label='PSI', ticks=ticks)

    # 绘制外壳
    theta = np.linspace(0, 2 * np.pi, 50)
    new_x_vals = np.linspace(z_min, z_max, 50)
    theta_grid, new_x_grid = np.meshgrid(theta, new_x_vals)
    new_y_cyl = R * np.cos(theta_grid)
    new_z_cyl = R * np.sin(theta_grid)
    ax.plot_wireframe(new_x_grid, new_y_cyl, new_z_cyl, color='gray', alpha=0.1)

    # 设置坐标轴标签和范围
    ax.set_xlabel('Z', labelpad=100)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('X', labelpad=10)
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(-R, R)
    ax.set_zlim(-R, R)
    ax.set_box_aspect([z_max - z_min, 2 * R, 2 * R])
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_yticks([-0.12, -0.06, 0, 0.06, 0.12])
    ax.set_zticks([-0.12, -0.06, 0, 0.06, 0.12])

    # 创建用于显示轨迹的空折线对象
    line, = ax.plot([], [], [], color='blue', lw=2, label='Trajectory')
    ax.legend(bbox_to_anchor=(1, 0.7))
    plt.tight_layout()

    # 准备逐帧写入GIF, 与功能2处理方式不同, 会在输出路径放置临时GIF，处理完每一帧会组合进GIF，防止爆内存
    total_points = len(new_x)
    output_filename = os.path.join(output_folder, "output_trajectory.gif")
    writer = imageio.get_writer(output_filename, fps=fps)
    print("如报告 dtype='uint8' 等 Matplotlib 字库问题, 请无视\nIgnore Matplotlib font problems such as dtype='uint8'")

    # 更新轨迹
    for i in range(1, total_points + 1, paint_step):
        line.set_data(new_x[:i], new_y[:i])
        line.set_3d_properties(new_z[:i])
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        writer.append_data(image)
        sys.stdout.write(f"\rProcessing : {i + paint_step - 1}/{total_points} ")       # 每处理一帧更新一次进度
        sys.stdout.flush()
    print("\nGIF Saving...")
    writer.close()
    print(f"GIF Saved to {output_filename}")
    plt.close(fig)

def generate_gif(file_list, output_filename, field_data=None, field_alpha=0.5, 
                R=0.12, z_min=0, z_max=1.7, x_columns=1, fps=10, read_step=1):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制背景场
    if field_data is not None:
        x_grid = np.linspace(z_min, z_max, field_data.shape[1])
        z_grid = np.linspace(-R, R, field_data.shape[0])
        X_mesh, Z_mesh = np.meshgrid(x_grid, z_grid)
        Y_mesh = np.zeros_like(X_mesh)
        cmap = plt.get_cmap('jet')
        norm = Normalize(vmin=field_data.min(), vmax=field_data.max())
        facecolors = cmap(norm(field_data))
        facecolors[..., 3] = field_alpha
        ax.plot_surface(X_mesh, Y_mesh, Z_mesh, rstride=1, cstride=1, facecolors=facecolors, shade=True)
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(field_data)
        ticks = np.linspace(field_data.min(), field_data.max(), 10)
        fig.colorbar(mappable, ax=ax, shrink=0.2, aspect=10, pad=0.1, label='PSI', ticks=ticks)

    # 绘制外壳
    theta = np.linspace(0, 2 * np.pi, 50)
    new_x_vals = np.linspace(z_min, z_max, 50)
    theta_grid, new_x_grid = np.meshgrid(theta, new_x_vals)
    new_y_cyl = R * np.cos(theta_grid)
    new_z_cyl = R * np.sin(theta_grid)
    ax.plot_wireframe(new_x_grid, new_y_cyl, new_z_cyl, color='gray', alpha=0.1)

    # 设置坐标轴标签和范围
    ax.set_xlabel('Z', labelpad=100)
    ax.set_ylabel('Y', labelpad=10)
    ax.set_zlabel('X', labelpad=10)
    ax.set_xlim(z_min, z_max)
    ax.set_ylim(-R, R)
    ax.set_zlim(-R, R)
    ax.set_box_aspect([z_max - z_min, 2 * R, 2 * R])
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_yticks([-0.12, -0.06, 0, 0.06, 0.12])
    ax.set_zticks([-0.12, -0.06, 0, 0.06, 0.12])

    # 绘制散点
    print("如报告 dtype='uint8' 等 Matplotlib 字库问题, 请无视\nIgnore Matplotlib font problems such as dtype='uint8'")
    scatter = ax.scatter([], [], [], c='blue', marker='o', s=20, label='Particles')
    ax.legend(bbox_to_anchor=(1, 0.7))
    plt.tight_layout()
    selected_files = file_list[read_step - 1::read_step]
    frames = []         # 用于保存每一帧的图像
    total = len(selected_files)
    for i, file in enumerate(selected_files):
        sys.stdout.write(f"\rProcessing : {os.path.basename(file)} ({i + 1}/{total}) ")
        sys.stdout.flush()
        x, y, z = load_data(file, x_columns)
        new_x = z
        new_y = y
        new_z = x
        scatter._offsets3d = (new_x, new_y, new_z)      # 更新 scatter 对象的数据

        # 绘制画布并转换为图像数组
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(image)

    print("\nGIF Saving...")
    output_filename = os.path.join(output_filename, "output_beam.gif")
    imageio.mimsave(output_filename, frames, fps=fps)
    print(f"GIF Saved to {output_filename}")
    plt.close(fig)


def main():
    config = Config()
    print("###############################")
    plt.rcParams['figure.dpi'] = config.plt_dpi
    field_data = None
    if config.add_field:
        print("Importing background field data files:")
        field_data = load_field(config.field_filename, *config.field_shape)
    if config.spot_open:
        print("Import particle spot data file:")
        plot_spot_distribution(config)
        
    if config.paradigm in [0, 1]:
        if config.paradigm == 0:
            print("Display static image of individual particle orbits:")
        if config.paradigm == 1:
            print("Displays static image of the scattered beam flow of multiple particles:")
        x, y, z = load_data(config.particle_filename, config.x_columns)
        plot_particles(x, y, z, config.paradigm, field_data=field_data, field_alpha=config.field_alpha)
    elif config.paradigm == 2:
        print("Display GIF image of individual particle orbits:")
        generate_trajectory_gif(config.particle_filename, config.output_path, field_data=field_data, field_alpha=config.field_alpha,
                              x_columns=config.x_columns, fps=config.fps, paint_step=config.paint_step)
    elif config.paradigm == 3:
        print("Displays GIF image of the scattered beam flow of multiple particles:")
        file_list = natsorted(glob.glob(os.path.join(config.particle_folder, "*.txt")))     # 获取所有 .txt 文件，按名称排序
        generate_gif(file_list, config.output_path, field_data=field_data, field_alpha=config.field_alpha, 
                    x_columns=config.x_columns, fps=config.fps, read_step=config.paint_step)
    else:
        print("Input the correct config.paradigm parameter")

if __name__ == "__main__":
    main()
