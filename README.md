# IMPLEMENTASI REINFORCEMENT LEARNING PADA ALGORITMA DYNAMIC WINDOW APPROACH (DWA) UNTUK NAVIGASI OTONOM ROBOT MOBILE HUSKY A200  

## Deskripsi
Project ini dibuat untuk memenuhi tugas skripsi dengan judul: "Implementasi Reinforcement learning pada Algoritma Dynamic Window Approach (DWA) untuk Navigasi Otonom Robot Mobile Husky A200"

Repository ini berisi:
1. **husky_custom/** → hasil modifikasi dari folder `husky_description` dan `husky_viz` pada package [Husky ROS](https://github.com/husky/husky).  
2. **husky_gym/** → package tambahan untuk eksperimen Reinforcement Learning (gym environment, launch, maps, dll.).

> ⚠️ Repo ini **bukan pengganti penuh** package `husky`.  
> Package asli tetap perlu di-clone, kemudian folder dalam repo ini digunakan untuk **replace** versi default.

## Cara Install & Run

### Prerequisites
- [Ubuntu 20.04](https://releases.ubuntu.com/20.04/)  
- [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu)  
- Catkin tools:
```bash
sudo apt install ros-noetic-catkin python3-catkin-tools 
```


### Langkah-langkah:
1. Setup Workspace:
```bash 
mkdir -p ~/husky_ws/src
cd ~/husky_ws/src
```
2. Clone Husky package:
```bash 
git clone https://github.com/husky/husky.git
```
3. Install dependencies Husky:
```bash
cd ~/husky_ws
rosdep install --from-paths src --ignore-src -r -y
```
4. Clone repo ini
```bash
git clone https://github.com/KyrieleisonFrans/repo-skripsi.git
```
5. Replace folder husky_description dan husky_viz dengan versi modifikasi:
```bash
rm -rf husky/husky_description husky/husky_viz
cp -r repo-skripsi/husky_custom/husky_description husky/
cp -r repo-skripsi/husky_custom/husky_viz husky/
```
6. Pindahkan file husky_gym keluar
```bash
mv repo-skripsi/husky_gym .
```
7. Hapus folder repo-skripsi
```bash
rm -rf repo-skripsi
```
8. Install dependencies:
```bash
cd ~/husky_ws
rosdep install --from-paths src --ignore-src -r -y
```
9. Build Workspace:
```bash
cd ~/husky_ws
catkin_make
```
10. Source workspace:
```bash
source devel/setup.bash
```



## Struktur Project
```bash
<repo-skripsi>/     
    ├── husky_custom/
    │   ├── husky_description/
    │   └── husky_viz/
    ├── husky_gym/
    │   ├── include/
    │   ├── launch/
    │   ├── map/
    │   ├── src/
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   └── setup.py
    └── README.md
```


## Kontak
Nama: Kyrieleison Charla Frans
NIM: 162112233078
Email: kyrieleison.charla.frans-2021@ftmm.unair.ac.id
&emsp;&emsp;&emsp; kyrieleison.frans@gmail.com

_Dibuat untuk keperluan skripsi - [2025]_