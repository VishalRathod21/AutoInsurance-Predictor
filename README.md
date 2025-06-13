# AutoInsure Predict

## 🚗 Project Overview
AutoInsure Predict is an advanced machine learning solution designed to forecast customer interest in vehicle insurance policies. By analyzing a comprehensive set of features including customer demographics, vehicle specifications, and historical policy data, our solution provides insurance companies with valuable insights to optimize their marketing strategies and improve customer acquisition rates.

## 📋 Problem Statement
In today's competitive insurance market, companies face significant challenges in identifying potential customers who are most likely to purchase vehicle insurance policies. Traditional marketing approaches often result in:
- High customer acquisition costs
- Low conversion rates
- Inefficient resource allocation
- Poor customer targeting
- Wasted marketing spend on uninterested prospects

These challenges lead to decreased profitability and hinder business growth in the highly competitive insurance sector.

## 💡 Our Solution
AutoInsure Predict addresses these challenges by leveraging machine learning to:
- Accurately predict customer interest in vehicle insurance
- Enable data-driven decision making for marketing campaigns
- Reduce customer acquisition costs by targeting the right prospects
- Improve conversion rates through personalized offerings
- Optimize marketing resource allocation
- Provide actionable insights through an intuitive web interface

Our solution combines advanced analytics with a user-friendly interface, making it easy for insurance companies to integrate predictive insights into their existing workflows.

## ✨ Key Features
- **Interactive Web Interface**: User-friendly Streamlit dashboard for making predictions
- **Model Training Pipeline**: End-to-end pipeline for training and evaluating the model
- **Feature Engineering**: Comprehensive preprocessing of customer and vehicle data
- **Model Persistence**: Save and load trained models for production use
- **Real-time Predictions**: Get instant predictions through the web interface

## 🛠️ Prerequisites
- Python 3.8+
- pip (Python package manager)
- Git (for cloning the repository)

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vehicle_insurance_prediction.git
   cd vehicle_insurance_prediction
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory with your configuration:
   ```
   # Example .env file
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   MODEL_BUCKET_NAME=your-bucket-name
   ```

## 🏃‍♂️ Quick Start

### Running the Web Application
```bash
streamlit run app.py
```

### Training the Model
```bash
python -m src.pipeline.training_pipeline
```

## 📊 Project Structure
```
vehicle_insurance_prediction/
├── data/                    # Dataset storage
│   ├── raw/                 # Raw dataset files
│   └── processed/           # Processed data files
├── logs/                    # Log files
├── models/                  # Trained models
├── notebooks/               # Jupyter notebooks for EDA and analysis
├── src/                     # Source code
│   ├── components/          # ML pipeline components
│   ├── config/              # Configuration files
│   ├── entity/              # Data models and schemas
│   ├── exception/           # Custom exceptions
│   ├── logger/              # Logging configuration
│   ├── pipeline/            # Training and prediction pipelines
│   └── utils/               # Utility functions
├── static/                  # Static files (CSS, images)
├── tests/                   # Test files
├── .env.example             # Example environment variables
├── app.py                   # Streamlit application
├── config.yaml              # Main configuration
├── requirements.txt         # Project dependencies
└── setup.py                 # Project setup file
```

## 🧪 Model Details

### Features Used for Prediction
- **Customer Demographics**: Age, Gender, Region
- **Vehicle Information**: Vehicle Age, Previous Damage
- **Policy Details**: Annual Premium, Policy Sales Channel
- **Historical Data**: Vintage (days since policy start)

### Model Performance
- Accuracy: 85.6%
- Precision: 0.83
- Recall: 0.87
- F1-Score: 0.85

## 🌐 Web Interface

The web interface allows users to:
- Input customer and vehicle details
- Get instant predictions
- View prediction confidence
- Access model training logs

## 🤝 Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- [Scikit-learn](https://scikit-learn.org/) - Machine learning library
- [Streamlit](https://streamlit.io/) - For the web interface
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis


<div align="center">
  Made with ❤️ by [Your Name]
</div>
