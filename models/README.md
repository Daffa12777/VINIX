# Folder Model

Letakkan file bobot hasil training di sini:

```
models/best.pt
```

File `best.pt` dihasilkan dari notebook `road_damage_yolo_training.ipynb`
(sel **Export Model for Deployment**). Tanpa file ini, aplikasi akan
menampilkan pesan error dan berhenti.

> `best.pt` sengaja di-*ignore* oleh git (lihat `.gitignore`) karena
> ukurannya besar. Saat deploy ke Streamlit Cloud, unggah lewat GitHub
> Release / Git LFS, atau commit manual bila ukuran memungkinkan.
