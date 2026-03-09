# 🔬 DataSense — Intelligent Dataset Analyzer & Auto-Cleaning System

[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-red)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**A powerful, AI-driven web application for comprehensive dataset analysis, quality assessment, and intelligent data cleaning. Built with Streamlit, DataSense automates the entire data preprocessing workflow with advanced visualization and reporting capabilities.**

## ✨ Key Features

### 📊 **Data Analysis & Profiling**
- **Automated Data Profiling**: Comprehensive statistical summaries, data type detection, and distribution analysis
- **Missing Value Analysis**: Intelligent detection and visualization of missing data patterns
- **Duplicate Detection**: Advanced algorithms to identify and analyze duplicate records
- **Outlier Detection**: Multiple methods (IQR, Z-score, Isolation Forest) for anomaly identification
- **Data Quality Scoring**: Automated health score calculation with actionable insights

### 🧹 **Intelligent Data Cleaning**
- **Smart Cleaning Strategies**: Context-aware suggestions based on data characteristics
- **Automated Cleaning Pipeline**: One-click application of multiple cleaning techniques
- **Before/After Comparisons**: Visual validation of cleaning operations
- **Customizable Cleaning Rules**: Fine-tune cleaning parameters based on domain knowledge

### 📈 **Advanced Visualizations**
- **Interactive Charts**: Plotly-powered visualizations for data exploration
- **Missing Data Heatmaps**: Pattern analysis for missing values
- **Correlation Matrices**: Feature relationship analysis
- **Distribution Plots**: Histograms, box plots, and categorical distributions
- **Drift Analysis**: Compare original vs cleaned datasets

### 🤖 **AI-Powered Insights**
- **Feature Importance**: ML-driven identification of significant variables
- **Pattern Recognition**: Automated discovery of data patterns and anomalies
- **Natural Language Reports**: AI-generated insights and recommendations
- **Query Assistant**: Natural language interface for data questions

### 📋 **Reporting & Documentation**
- **Automated Report Generation**: Comprehensive analysis reports in multiple formats
- **Export Options**: PDF, CSV, and text format exports
- **Documentation Creation**: Auto-generated data dictionaries and cleaning logs
- **Audit Trail**: Complete tracking of all cleaning operations

## 🚀 Quick Start

### Prerequisites
- **Python 3.8 or higher**
- **pip package manager**

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/JAYESHX19/dataset_analyzer.git
   cd dataset_analyzer
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:8501`

## 📁 Project Structure

```
dataset_analyzer/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── modules/                   # Core functionality modules
│   ├── data_loader.py        # Dataset loading and metadata
│   ├── profiling.py          # Data profiling and statistics
│   ├── quality_detection.py  # Data quality assessment
│   ├── cleaning_engine.py    # Data cleaning algorithms
│   ├── visualization_engine.py # Chart generation
│   ├── insight_engine.py     # AI-powered insights
│   ├── advanced_analysis.py  # Comparative analysis
│   ├── report_generator.py   # Report generation
│   └── query_assistant.py    # NLP query interface
├── utils/                     # Helper utilities
│   ├── helpers.py           # Common utility functions
│   └── __init__.py
├── assets/                    # Sample datasets
├── cleaned_data/             # Processed datasets
└── reports/                  # Generated reports
```

## 🎯 How to Use

### 1. **Upload Your Dataset**
- **Supported formats**: CSV, Excel (XLSX, XLS), JSON
- **Maximum file size**: 100MB
- **Automatic data type detection**

### 2. **Data Profiling**
- View comprehensive statistics
- Analyze data distributions
- Identify potential issues

### 3. **Quality Assessment**
- Check data quality score
- Identify missing values, duplicates, outliers
- Get cleaning recommendations

### 4. **Data Cleaning**
- Choose from suggested cleaning strategies
- Customize cleaning parameters
- Apply automated cleaning pipeline

### 5. **Analysis & Insights**
- Explore visualizations
- Generate insights and reports
- Export cleaned data

## 🔧 Advanced Features

### **Query Assistant**
Ask questions about your data in natural language:
- "What are the top 5 most correlated features?"
- "Show me rows with missing values in column X"
- "What's the average value of column Y?"

### **Batch Processing**
- Process multiple datasets simultaneously
- Apply consistent cleaning across files
- Generate comparative reports

### **Custom Cleaning Rules**
- Define domain-specific cleaning logic
- Create reusable cleaning templates
- Save and share cleaning workflows

## 📊 Supported Data Types

### **File Formats**
- **CSV** (comma, semicolon, tab delimited)
- **Excel files** (.xlsx, .xls)
- **JSON files**
- **Text files** (delimited)

### **Data Types**
- **Numeric** (integers, floats)
- **Categorical** (nominal, ordinal)
- **Text/String** data
- **Date/Time** values
- **Boolean** values

## 🛠️ Configuration

### **Environment Variables**
Create a `.env` file for custom settings:
```env
MAX_FILE_SIZE=100000000
DEFAULT_ENCODING=utf-8
CHUNK_SIZE=10000
ENABLE_GPU=false
```

### **Custom Settings**
Modify `utils/config.py` for:
- Default visualization themes
- Cleaning algorithm parameters
- Report templates

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### **Development Guidelines**
- Follow **PEP 8** for Python code style
- Add **docstrings** to new functions
- Include **unit tests** for new features
- Update **documentation** as needed

## 🐛 Troubleshooting

### **Common Issues**

**Q: Large files cause memory errors**
A: Enable **chunked processing** in settings or reduce file size

**Q: Missing dependencies**
A: Run `pip install -r requirements.txt` with admin privileges

**Q: Visualization not loading**
A: Check browser console for JavaScript errors

**Q: Export fails**
A: Ensure write permissions in reports directory

### **Performance Tips**
- Use **chunked processing** for large datasets
- Clear **cache regularly**
- Disable **unused features** in settings

## 📝 API Reference

### **Core Modules**

#### **Data Loader**
```python
from modules.data_loader import load_dataset
df = load_dataset('file.csv', encoding='utf-8')
```

#### **Quality Detection**
```python
from modules.quality_detection import compute_health_score
score = compute_health_score(df)
```

#### **Cleaning Engine**
```python
from modules.cleaning_engine import apply_cleaning
cleaned_df = apply_cleaning(df, strategy='auto')
```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** - For the amazing web app framework
- **Plotly** - For interactive visualizations
- **Pandas** - For powerful data manipulation
- **Scikit-learn** - For machine learning algorithms
- **Data Science Community** - For inspiration and feedback

## 📞 Support

- 📧 **Email**: jayeshppatel00@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/JAYESHX19/dataset_analyzer/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/JAYESHX19/dataset_analyzer/discussions)

## 🔄 Version History

- **v1.0.0** - Initial release with core functionality
- **v1.1.0** - Added query assistant and advanced analytics
- **v1.2.0** - Enhanced visualization and reporting

---

**Made with ❤️ by [Jayesh Patel](https://github.com/JAYESHX19)**

*Transform your data cleaning workflow with AI-powered automation!*
