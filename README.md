# 🧠 Brain Tumor Detection and Visualization 🖥️

## 🌟 Overview

This cutting-edge project implements a 3D semantic segmentation system for detecting and visualizing brain tumors in MRI scans. Our system processes MRI data in real-time, segments different types of brain tissues, and provides state-of-the-art visualization methods to aid in tumor analysis.

## 🚀 Features

- 🔬 3D semantic segmentation of brain MRI scans
- ⚡ Real-time processing of MRI data
- 🎨 Multiple visualization methods:
  - 📊 Multi-slice 2D views
  - 🌐 3D surface renderings of brain and tumor
  - 🎭 Color-coded segmentation of different tissue types
  - 🖱️ Interactive web-based 3D visualizations
  - 📈 Nilearn-based plots (anatomical, EPI, ROI)
- 📏 Calculation of tumor volume and dimensions

## 🛠️ Requirements

- Python 3.7+
- NumPy
- Matplotlib
- Nibabel
- Plotly
- Scikit-image
- Nilearn
- Flask

## 📥 Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/brain-tumor-detection.git
   cd brain-tumor-detection
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## 🏃‍♂️ Usage

1. Start the Flask server:
   ```
   python app1.py
   ```

2. Open a web browser and navigate to `http://localhost:5000`

3. Use the intuitive web interface to select a Patient from the Patients Page

## 📁 File Structure

- `app.py`: Main Flask application
- `templates/`: HTML templates for the web interface
- `static/`: Static files (CSS, JavaScript)
- `requirements.txt`: List of Python dependencies

## 🔍 How It Works

1. 📥 The system loads MRI data using Nibabel
2. 🧼 Preprocessing is applied to normalize the image data
3. 🧠 3D semantic segmentation identifies different tissue types
4. 🎨 Various visualization methods are applied to the segmented data
5. 🖥️ Results are displayed through an interactive web interface

## 📊 Output Examples

![](https://github.com/tejash09/Brain-tumor/blob/main/img/test_gif_BraTS20_Training_001_flair.gif)

![](https://github.com/tejash09/Brain-tumor/blob/main/img/newplot.png)
![](https://github.com/tejash09/Brain-tumor/blob/main/img/slice.png)
![](https://github.com/tejash09/Brain-tumor/blob/main/img/hemisperical%20view.png)
![](https://github.com/tejash09/Brain-tumor/blob/main/img/Figure_1.png)
## 🤝 Contributing

We welcome contributions to this project! Please fork the repository and submit a pull request with your innovative changes.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👏 Acknowledgments

- 🙏 Thanks to the Nilearn and Plotly teams for their excellent visualization libraries
- 💡 This project was inspired by groundbreaking advances in medical imaging and deep learning for tumor detection

## 🔮 Future Enhancements

- Integration with cloud storage for seamless data management
- Implementation of AI-driven prognostic features
- Development of a mobile app for on-the-go access to visualizations

Stay tuned for more exciting updates! 🚀🧠
