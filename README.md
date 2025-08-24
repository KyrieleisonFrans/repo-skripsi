# IMPLEMENTASI REINFORCEMENT LEARNING PADA ALGORITMA DYNAMIC WINDOW APPROACH (DWA) UNTUK NAVIGASI OTONOM ROBOT MOBILE HUSKY A200  

## Deskripsi
Project ini dibuat untuk memenuhi tugas skripsi dengan judul: "Implementasi Reinforcement learning pada Algoritma Dynamic Window Approach (DWA) untuk Navigasi Otonom Robot Mobile Husky A200"

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
git clone https://github.com/username/repo-skripsi.git 
```
5. Replace folder husky_description dan husky_viz dengan versi modifikasi:
```bash
cp -r husky_custom/husky_description husky/
cp -r husky_custom/husky_viz husky/
```


## Struktur Project



## Kontak
Nama : Kyrieleison Charla Frans
NIM : 162112233078
Email: kyrieleison.charla.frans-2021@ftmm.unair.ac.id
kyrieleison.frans@gmail.com

