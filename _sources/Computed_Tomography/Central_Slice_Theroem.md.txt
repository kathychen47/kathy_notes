
# Central Slice Theorem

It states that the one-dimensional Fourier transformer of the projection of a function or $image(x,y)$ along a direction $\theta$, is equal to a vertical slice two dimensional Fourier transform $F(\omega_1, \omega_2)$ of $f(x,y)$ passing through the origin and oriented at an angle $\theta$ with respect to the $$\omega_1$$ axis. 



Assume $t\text{-}s$ is a rotated coordinate system according to $x\text{-}y$. 

Slice $f(x,y)$ in $x\text{-}y$ coordinate is shown as $f^{'}(t,s)$ in $t\text{-}s$ coordinate system.

$$ f^{'}(t,s) = \int_0^\pi  {d\theta } \int_{ - \infty }^{ + \infty } {P(\omega ,\theta )\left| \omega  \right|} {e^{2\pi \omega t}}d\omega $$

# FBP







## Codes for FBP

Tomographic reconstruction steps:



根据代码中的 `print(proj.shape, flat.shape, dark.shape, theta.shape)`，四个参数的含义如下：

- `proj`: 形状为 `(num_angles, num_channels, num_slices)` 的三维 numpy 数组，表示待重构的投影数据。其中，`num_angles` 是投影角度数，`num_channels` 是每个角度下的通道数（也可以称为像素数），`num_slices` 是待重构的切片数（即待重构的三维物体的层数）。因为代码中使用了 `sino=(0, 2)` 参数，所以只选择了两个投影角度（即 `num_angles=2`），这也导致了 `proj` 的形状是 `(2, num_channels, num_slices)`。因为 `proj` 是待重构的投影数据，所以每个像素点的数值代表了射线在穿过物体时的衰减量，可以用于重构三维物体。

- `flat`: 形状与 `proj` 相同的三维 numpy 数组，表示用于校准的 flat field 数据。因为校准的 flat field 数据和待重构的投影数据的形状必须相同，所以 `flat` 的形状也是 `(num_angles, num_channels, num_slices)`。

- `dark`: 形状与 `proj` 相同的三维 numpy 数组，表示用于校准的 dark field 数据。因为校准的 dark field 数据和待重构的投影数据的形状必须相同，所以 `dark` 的形状也是 `(num_angles, num_channels, num_slices)`。dark field 数据一般用于校准探测器的漂移和噪声等问题。

- `theta`: 形状为 `(num_angles,)` 的一维 numpy 数组，表示所有投影角度的值。因为在代码中使用了 `sino=(0, 2)` 参数，所以只选择了两个投影角度，即 `theta` 的长度是 2。`theta` 一般用于控制投影角度，可以通过改变 `theta` 的值来获取不同的投影数据，从而用于重构不同的三维物体。

- 

- According to the code `print(proj.shape, flat.shape, dark.shape, theta.shape)`, the four parameters mean the following:

  - `proj`: a three-dimensional numpy array of shape `(num_angles, num_channels, num_slices)`, representing the projection data to be reconstructed. Here, `num_angles` is the number of projection angles, `num_channels` is the number of pixels per angle (also known as the number of detector channels), and `num_slices` is the number of slices to be reconstructed (i.e., the number of layers of the three-dimensional object to be reconstructed). Because `sino=(0, 2)` is used in the code, only two projection angles are selected (i.e., `num_angles=2`), which leads to the shape of `proj` being `(2, num_channels, num_slices)`. As the projection data to be reconstructed, the value of each pixel represents the attenuation of the X-ray beam when passing through the object, which can be used to reconstruct the three-dimensional object.
  - `flat`: a three-dimensional numpy array of the same shape as `proj`, representing the flat field data used for calibration. Because the shape of the flat field data used for calibration must be the same as that of the projection data to be reconstructed, the shape of `flat` is also `(num_angles, num_channels, num_slices)`.
  - `dark`: a three-dimensional numpy array of the same shape as `proj`, representing the dark field data used for calibration. Because the shape of the dark field data used for calibration must be the same as that of the projection data to be reconstructed, the shape of `dark` is also `(num_angles, num_channels, num_slices)`. The dark field data is generally used to correct for detector drift and noise.
  - `theta`: a one-dimensional numpy array of shape `(num_angles,)`, representing the values of all projection angles. Because `sino=(0, 2)` is used in the code, only two projection angles are selected, so the length of `theta` is 2. `theta` is generally used to control the projection angles and can be modified to obtain different projection data for reconstructing different three-dimensional objects.

- 

  ```python
  ```

  



```python
import numpy as np
import h5py

# Load the data
with h5py.File('tooth.h5', 'r') as f:
    proj = f['exchange/data'][sino[0]:sino[1]+1].astype(np.float32)
    flat = f['exchange/data_white'][()].astype(np.float32)
    dark = f['exchange/data_dark'][()].astype(np.float32)
    theta = f['exchange/theta'][()].astype(np.float32)

# Normalize the data
proj = (proj - dark) / (flat - dark)

# Calculate the sinogram center
center = proj.shape[2] // 2

# Define the reconstruction parameters
num_slices = proj.shape[1]
num_cols = proj.shape[2]
num_rows = num_cols

# Initialize the reconstruction volume
recon = np.zeros((num_slices, num_rows, num_cols), dtype=np.float32)

# Perform the backprojection
for i in range(num_slices):
    print('Reconstructing slice {}/{}'.format(i+1, num_slices))
    for j in range(num_proj):
        # Calculate the projection angle
        ang = theta[j] * np.pi / 180.0

        # Calculate the projection offset
        offset = j - center

        # Calculate the backprojection angle
        bp_ang = np.arctan2(offset, center)

        # Calculate the distance from the center
        dist = np.sqrt(offset ** 2 + center ** 2)

        # Calculate the backprojection weight
        weight = np.sin(bp_ang) / dist

        # Backproject the projection data
        recon[i] += weight * proj[j, i]

# Normalize the reconstructed volume
recon /= num_proj

# Save the reconstructed volume
with h5py.File('recon.h5', 'w') as f:
    f.create_dataset('recon', data=recon, dtype=np.float32)
```

This code reads in the projection data (`proj`), flat field data (`flat`), dark field data (`dark`), and projection angles (`theta`) from the `tooth.h5` file, performs normalization on the projection data, and calculates the sinogram center. It then initializes the reconstruction volume (`recon`) with zeros and performs the back projection to reconstruct the CT images. Finally, it normalizes the reconstructed volume and saves it to a new `recon.h5` file.





```python
import numpy as np
import h5py
import matplotlib.pyplot as plt

def fbp_reconstruction(proj, flat, dark, theta):
    """
    使用中心极限定理实现FBP重建
    :param proj: 投影数据，形状为(n_theta, n_s)
    :param flat: 平坦场数据，形状为(n_theta, n_s)
    :param dark: 暗场数据，形状为(n_theta, n_s)
    :param theta: 投影角度，形状为(n_theta,)
    :return: 重建图像，形状为(n_s, n_s)
    """
    # 预处理平坦场和暗场数据
    flat = (flat - dark) / flat
    flat = np.log(flat)

    # 投影数据加上小常数，避免对数运算时出现无穷大
    proj = proj + 1e-12

    # 将投影数据转换为对数空间
    proj = np.log(proj)

    # 重建图像大小
    n_s = proj.shape[1]

    # 构建投影矩阵
    p = np.linspace(-0.5, 0.5, n_s+1)[:-1]
    proj_mat = np.abs(np.tile(p, (proj.shape[0], 1)) - np.tile(p, (proj.shape[0], 1)).T)
    proj_mat[proj_mat > 0.5] = 0.5
    proj_mat = 2 * np.arcsin(proj_mat / 0.5)

    # 进行重建
    reconstruction = np.zeros((n_s, n_s))
    for i in range(proj.shape[0]):
        # 使用中心极限定理计算傅里叶变换
        ft = np.fft.fftshift(np.fft.fft(proj[i]))
        # 滤波
        filtered_ft = ft * proj_mat[i]
        # 逆傅里叶变换
        filtered_proj = np.real(np.fft.ifft(np.fft.ifftshift(filtered_ft)))
        # 使用中心极限定理计算重建图像
        reconstruction += np.exp(filtered_proj) * np.sin(theta[i])

    # 根据投影角度进行归一化
    reconstruction /= np.sum(np.sin(theta))

    # 根据预处理的平坦场进行反投影校正
    for i in range(proj.shape[0]):
        ft = np.fft.fftshift(np.fft.fft(np.exp(proj[i] - flat[i])))
        filtered_ft = ft * proj_mat[i]
        filtered_proj = np.real(np.fft.ifft(np.fft.ifftshift(filtered_ft)))
        reconstruction -= filtered_proj * np.sin(theta[i])

    # 归一化
    reconstruction /= np.sum(proj_mat)

    return reconstruction

# 读取数据
with h5py.File('tooth.h5', 'r') as f:
    proj = f['exchange/data'][...]
    flat = f['exchange/data_white'][...]
    dark = f['exchange/data_dark'][...]
    theta = f['exchange/theta'][...]

# 选择部分投影数据进行重建
proj = proj[0:2, :]
# 进行重建
reconstruction = fbp_reconstruction(proj, flat, dark, theta)

# 显示重建图像
plt.imshow(reconstruction, cmap='gray')
plt.axis('off')
plt.show()
```



```python
pythonCopy codeimport numpy as np
import h5py

def fbp_reconstruction(proj, flat, dark, theta):
    """
    使用中心极限定理实现FBP重建
    :param proj: 投影数据，形状为(n_theta, n_s)
    :param flat: 平坦场数据，形状为(n_theta, n_s)
    :param dark: 暗场数据，形状为(n_theta, n_s)
    :param theta: 投影角度，形状为(n_theta,)
    :return: 重建图像，形状为(n_s, n_s)
    """
    # 预处理平坦场和暗场数据
    flat = (flat - dark) / flat
    flat = np.log(flat)

    # 投影数据加上小常数，避免对数运算时出现无穷大
    proj = proj + 1e-12

    # 将投影数据转换为对数空间
    proj = np.log(proj)

    # 重建图像大小
    n_s = proj.shape[1]

    # 构建投影矩阵
    p = np.linspace(-0.5, 0.5, n_s+1)[:-1]
    proj_mat = np.abs(np.tile(p, (proj.shape[0], 1)) - np.tile(p, (proj.shape[0], 1)).T)
    proj_mat[proj_mat > 0.5] = 0.5
    proj_mat = 2 * np.arcsin(proj_mat / 0.5)

    # 进行重建
    reconstruction = np.zeros((n_s, n_s))
    for i in range(proj.shape[0]):
        # 使用中心极限定理计算傅里叶变换
        ft = np.zeros((n_s,), dtype=np.complex128)
        for j in range(n_s):
            for k in range(n_s):
                ft[j] += proj[i, k] * np.exp(-2j * np.pi * j * k / n_s)

        # 滤波
        filtered_ft = np.zeros((n_s,), dtype=np.complex128)
        for j in range(n_s):
            for k in range(n_s):
                filtered_ft[j] += ft[k] * proj_mat[i, k-j+n_s//2]

        # 逆傅里叶变换
        filtered_proj = np.zeros((n_s,))
        for j in range(n_s):
            for k in range(n_s):
                filtered_proj[j] += filtered_ft[k] * np.exp(2j * np.pi * j * k / n_s)

        # 使用中心极限定理计算重建图像
        for j in range(n_s):
            for k in range(n_s):
                s = p[k]
                x = s * np.cos(theta[i])
                y = s * np.sin(theta[i])
                f = np.exp(filtered_proj[j])
                reconstruction[j, k] += f * np.sqrt((np.pi/2)/s) * np.exp(-(x**2+y**2)/(2*s))

    # 根据投影角度进行归一化
    reconstruction /= np.sum(np.sin(theta))

    return reconstruction
```



在这个版本的代码中，我添加



![img](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%2730%27%20height=%2730%27/%3e)![Kathy Chen](https://chat.openai.com/_next/image?url=https%3A%2F%2Flh3.googleusercontent.com%2Fa%2FAGNmyxaDYmERjMwydPr4csjMJ-5koyZ7nO9tH33CkB-F%3Ds96-c&w=64&q=75)

继续

![image-20230309180038525](https://lh3.googleusercontent.com/9cB7NiX0DCZbNPd1rrhlcVGwMTq5siTft5L3Ju8--bFsIBU8JeQjJ8vyLVQBbMO9u_B2o3HNvDklD9GHuKBFOEpWcYdicR7Rc1G2rD4V)



在上一个回答中，我添加了更详细的代码，包含了一些公式。以下是这个代码的解释：

1. `flat` 和 `dark` 数据进行预处理，使用以下公式将其转换为对数空间：

   ```
   makefileCopy codeflat = (flat - dark) / flat
   flat = np.log(flat)
   ```

2. 将投影数据转换为对数空间，使用以下代码：

   ```
   cCopy code
   proj = np.log(proj + 1e-12)
   ```

   注意，我们在投影数据上加上一个小常数，避免对数运算时出现无穷大。

3. 构建投影矩阵，使用以下公式：

   ```
   cssCopy codep = np.linspace(-0.5, 0.5, n_s+1)[:-1]
   proj_mat = np.abs(np.tile(p, (proj.shape[0], 1)) - np.tile(p, (proj.shape[0], 1)).T)
   proj_mat[proj_mat > 0.5] = 0.5
   proj_mat = 2 * np.arcsin(proj_mat / 0.5)
   ```

   这里 `p` 是一个均匀分布的采样点，投影矩阵可以使用 `proj_mat` 计算，这个矩阵定义了在傅里叶空间内的滤波操作。

4. 对每个投影进行重建，使用以下步骤：

   a. 使用中心极限定理计算傅里叶变换，使用以下代码：

   ```
   scssCopy codeft = np.zeros((n_s,), dtype=np.complex128)
   for j in range(n_s):
       for k in range(n_s):
           ft[j] += proj[i, k] * np.exp(-2j * np.pi * j * k / n_s)
   ```

   b. 滤波傅里叶变换，使用以下代码：

   ```
   goCopy codefiltered_ft = np.zeros((n_s,), dtype=np.complex128)
   for j in range(n_s):
       for k in range(n_s):
           filtered_ft[j] += ft[k] * proj_mat[i, k-j+n_s//2]
   ```

   c. 使用中心极限定理计算逆傅里叶变换，使用以下代码：

   ```
   scssCopy codefiltered_proj = np.zeros((n_s,))
   for j in range(n_s):
       for k in range(n_s):
           filtered_proj[j] += filtered_ft[k] * np.exp(2j * np.pi * j * k / n_s)
   ```

   d. 使用中心极限定理计算重建图像，使用以下代码：

   ```
   scssCopy codefor j in range(n_s):
       for k in range(n_s):
           s = p[k]
           x = s * np.cos(theta[i])
           y = s * np.sin(theta[i])
           f = np.exp(filtered_proj[j])
           reconstruction[j, k] += f * np.sqrt((np.pi/2)/s) * np.exp(-(x**2+y**2)/(2*s))
   ```

5. 根据投影角度进行归一化，使用以下代码：

   ```
   sqlCopy code
   reconstruction /= np.sum(np.sin(theta))
   ```

以上就是更详细的代码解释。



![img](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%2730%27%20height=%2730%27/%3e)![Kathy Chen](https://chat.openai.com/_next/image?url=https%3A%2F%2Flh3.googleusercontent.com%2Fa%2FAGNmyxaDYmERjMwydPr4csjMJ-5koyZ7nO9tH33CkB-F%3Ds96-c&w=64&q=75)

可以再加一个可视化代码吗





当然可以，以下是一个简单的可视化代码，用于显示重建图像：

```python
pythonCopy codeimport matplotlib.pyplot as plt

plt.imshow(reconstruction, cmap='gray')
plt.axis('off')
plt.show()
```

这个代码使用 Matplotlib 库绘制灰度图像。`cmap='gray'` 参数指定使用灰度颜色映射，`plt.axis('off')` 参数关闭图像坐标轴。如果你想将图像保存到文件，可以使用以下代码：

```
pythonCopy code
plt.savefig('reconstruction.png', dpi=300, bbox_inches='tight')
```

其中，`dpi=300` 指定保存图像的分辨率为 300 dpi，`bbox_inches='tight'` 用于裁剪图像边缘。













CT重建是一种将X射线投影转换为三维图像的过程。其中一种常用的重建算法是滤波反投影算法（FBP）。在此，我们将演示如何使用中心极限定理（Central Limit Theorem）和FBP算法重建CT图像。

由于题目要求不使用任何包，我们需要手动实现CT重建过程。首先，我们读取数据集并进行预处理。我们首先对每个投影减去它对应的暗场（dark field），然后将每个投影减去对应的平场（flat field）并除以平场。

```
pythonCopy code# 读取数据集并进行预处理
import h5py

with h5py.File('tooth.h5', 'r') as f:
    proj = f['exchange/data'][0:2]
    flat = f['exchange/data_white'][0]
    dark = f['exchange/data_dark'][0]
    theta = f['exchange/theta'][:]
    
proj -= dark
proj = (proj - flat) / flat
```

接下来，我们需要实现一个函数，将一维投影数据集proj_i通过逆Radon变换重建为二维图像。在此，我们将使用中心极限定理来平滑投影数据集并减少噪声。

```
pythonCopy code# 中心极限定理
def central_limit_theorem(proj_i):
    n = proj_i.shape[0]
    m = 10*n
    
    # 采样一些子集
    samples = np.random.choice(proj_i, (m, n))
    
    # 计算样本均值和标准差
    sample_mean = np.mean(samples, axis=1)
    sample_std = np.std(samples, axis=1, ddof=1)
    
    # 标准化子集
    samples_norm = (samples - sample_mean[:, np.newaxis]) / sample_std[:, np.newaxis]
    
    # 将子集平均值加权平均
    weights = 1 / (sample_std**2)
    weights_sum = np.sum(weights)
    proj_i_smoothed = np.sum(weights[:, np.newaxis] * samples_norm, axis=0) / weights_sum
    
    return proj_i_smoothed

# FBP CT重建算法
def fbp(proj, theta):
    # 对每个投影进行中心极限定理平滑
    proj_smoothed = np.zeros_like(proj)
    for i in range(proj.shape[0]):
        proj_smoothed[i, :] = central_limit_theorem(proj[i, :])
    
    # 对每个投影进行雷登变换
    sinogram = np.apply_along_axis(np.fft.fft, 0, proj_smoothed)

    # 应用RAMP滤波器
    ramp_filter = np.abs(np.linspace(-np.pi / 2, np.pi / 2, proj.shape[1], endpoint=False))
    filtered_sinogram = sinogram * ramp_filter[:, np.newaxis, np.newaxis]

    # 反变换得到重建图像
    filtered_proj = np.apply_along_axis(np.fft.ifft, 0, filtered_sinogram).real
    return np.apply_along_axis(lambda p: np.interp(theta, np.linspace(0, 180, proj.shape
```



![img](data:image/svg+xml,%3csvg%20xmlns=%27http://www.w3.org/2000/svg%27%20version=%271.1%27%20width=%2730%27%20height=%2730%27/%3e)![Kathy Chen](https://chat.openai.com/_next/image?url=https%3A%2F%2Flh3.googleusercontent.com%2Fa%2FAGNmyxaDYmERjMwydPr4csjMJ-5koyZ7nO9tH33CkB-F%3Ds96-c&w=64&q=75)

继续





# 对每个投影进行中心极限定理平滑

proj_smoothed = np.zeros_like(proj) for i in range(proj.shape[0]): proj_smoothed[i, :] = central_limit_theorem(proj[i, :])

# 对每个投影进行雷登变换

sinogram = np.apply_along_axis(np.fft.fft, 0, proj_smoothed)

# 应用RAMP滤波器

ramp_filter = np.abs(np.linspace(-np.pi / 2, np.pi / 2, proj.shape[1], endpoint=False)) filtered_sinogram = sinogram * ramp_filter[:, np.newaxis, np.newaxis]

# 反变换得到重建图像

filtered_proj = np.apply_along_axis(np.fft.ifft, 0, filtered_sinogram).real recon = fbp(filtered_proj, theta)

# 显示重建图像

import matplotlib.pyplot as plt from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(figsize=(6, 6)) grid = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=0.1)

grid[0].imshow(recon, cmap='gray') grid[0].set_title('Reconstructed image')

plt.show()







