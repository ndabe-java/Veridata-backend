import pandas as pd
import numpy as np
import io
from collections import defaultdict
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
 
# =========================================================
# HELPER FUNCTION TO CONVERT NUMPY TYPES TO NATIVE PYTHON TYPES
# =========================================================
def convert_numpy_to_native(obj):
    """
    Recursively converts numpy types (int64, float64) within a dictionary
    or list to standard Python types for JSON serialization.
    """
    if isinstance(obj, (np.generic, np.number)):
        # Convert single numpy number to native Python type (e.g., np.int64 -> int)
        return obj.item()
    elif isinstance(obj, dict):
        # Recursively process dictionary values
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, np.ndarray)):
        # Recursively process list or array elements
        return [convert_numpy_to_native(i) for i in obj]
    else:
        return obj
 
# =========================================================
# ANALYSIS FUNCTION
# =========================================================
 
def analyze_dataset_expanded(df: pd.DataFrame) -> dict:
    """
    Expanded analysis for the Data Nutrition Label, including statistical checks.
    """
    analysis_results = {'Global Metrics': {}, 'Column Metrics': defaultdict(dict), 'Advanced Checks': {}}
 
    # --- A. Global Dataset Metrics ---
    # Convert results immediately to Python int/float to avoid later issues
    analysis_results['Global Metrics'] = {
        'Total Rows': int(len(df)),
        'Total Columns': int(len(df.columns)),
        # Ensure percentages are converted to standard float
        'Duplicate Rows (%)': float((df.duplicated().sum() / len(df)) * 100),
        'Missing Values (Total %)': float((df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
    }
 
    # --- B. Column-Level Metrics ---
    column_metrics = analysis_results['Column Metrics']
 
    for col in df.columns:
        series = df[col]
        unique_count = series.nunique()
        
        column_metrics[col]['Data Type'] = str(series.dtype)
        # Convert all calculated numerical results to float
        column_metrics[col]['Missing Values (%)'] = float((series.isnull().sum() / len(series)) * 100)
        column_metrics[col]['Unique Values'] = int(unique_count)
        column_metrics[col]['Cardinality (%)'] = float((unique_count / len(series)) * 100)
 
        # Numerical Analysis
        if np.issubdtype(series.dtype, np.number):
            # Numerical stats (converted to standard float)
            column_metrics[col]['Mean'] = float(series.mean())
            column_metrics[col]['Median'] = float(series.median())
            column_metrics[col]['Std Dev'] = float(series.std())
            column_metrics[col]['Min'] = convert_numpy_to_native(series.min()) # Use helper for min/max
            column_metrics[col]['Max'] = convert_numpy_to_native(series.max())
            
            # Outlier Detection (IQR)
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers_count = len(series[(series < lower_bound) | (series > upper_bound)])
            column_metrics[col]['Outliers (IQR) %'] = float((outliers_count / len(series)) * 100)
            
        # Categorical/Low Cardinality Analysis
        elif series.dtype == 'object' or unique_count < 20:
            # The .to_dict() creates standard Python types, but we'll run the conversion helper later just in case
            top_counts = series.value_counts(normalize=True).head(5)
            column_metrics[col]['Top 5 Values (Pct)'] = {k: f"{v * 100:.2f}%" for k, v in top_counts.to_dict().items()}
        
        # Status Flags (unchanged)
        if unique_count < 5 or (np.issubdtype(series.dtype, np.number) and series.std() == 0):
             column_metrics[col]['**Status Flag**'] = 'Constant/Near-Constant'
        elif column_metrics[col]['Missing Values (%)'] > 90:
             column_metrics[col]['**Status Flag**'] = 'Mostly Missing'
        else:
             column_metrics[col]['**Status Flag**'] = 'OK'
 
 
    # --- C. Advanced Global Checks ---
    
    # High Correlation Check (The logic produces Python lists/dicts, but we apply the final conversion)
    corr_matrix = df.select_dtypes(include=np.number).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    highly_correlated_pairs = []
    for i in range(len(upper.columns)):
        for j in range(i):
            if upper.iloc[i, j] > 0.90:
                highly_correlated_pairs.append({
                    'Feature 1': upper.columns[j],
                    'Feature 2': upper.index[i],
                    'Correlation': f"{upper.iloc[i, j]:.2f}"
                })
 
    analysis_results['Advanced Checks']['Highly Correlated Pairs (> 0.90)'] = highly_correlated_pairs
 
    # Data Drift Proxy (Start vs. End Mean)
    # The logic here already converts to strings (f"{percent_change:.2f}%")
    # ... (Drift logic unchanged) ...
    split_point_1 = int(len(df) * 0.2)
    split_point_2 = int(len(df) * 0.8)
    drift_metrics = {}
    
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.number) and len(df) > 100:
            mean_start = df[col].iloc[:split_point_1].mean()
            mean_end = df[col].iloc[split_point_2:].mean()
            
            if pd.notna(mean_start) and pd.notna(mean_end) and mean_start != 0:
                percent_change = (abs(mean_end - mean_start) / mean_start) * 100
                if percent_change > 10.0:
                    drift_metrics[col] = f"{percent_change:.2f}%"
                
    analysis_results['Advanced Checks']['Mean Drift (Start vs End > 10% Change)'] = drift_metrics
 
    # =========================================================
    # CRITICAL FINAL STEP: APPLY RECURSIVE CONVERSION
    # =========================================================
    final_results = convert_numpy_to_native(analysis_results)
    return final_results # Return the fully converted dictionary
 
# =========================================================
# PDF GENERATION FUNCTION (Unchanged from previous versions)
# =========================================================
 
def generate_pdf_label(analysis_data: dict, dataset_name: str) -> io.BytesIO:
    # ... (Keep this function as is, since the input data is now clean Python types)
    # ... (I will not paste the full function here to keep the answer concise)
    
    # To run your uvicorn server, make sure you've saved the entire contents of
    # generate_pdf_label from the previous correct response into label_generator.py.
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                            title=f"Data Nutrition Label: {dataset_name}")
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    elements.append(Paragraph(f"Data Nutrition Label: {dataset_name}", styles['Title']))
    elements.append(Spacer(1, 18))
    
    # ------------------ A. Global Health Check ------------------
    elements.append(Paragraph("A. Global Health Check 🩺", styles['h2']))
    
    global_metrics = analysis_data.get('Global Metrics', {})
    global_data = [
        ["Metric", "Value", "Flag"],
        ["Total Rows", f"{global_metrics.get('Total Rows', 0):,}", ""],
        ["Total Columns", str(global_metrics.get('Total Columns', 0)), ""],
        ["Duplicate Rows (%)", f"{global_metrics.get('Duplicate Rows (%)', 0):.2f}%", "HIGH" if global_metrics.get('Duplicate Rows (%)', 0) > 5 else "OK"],
        ["Missing Values (Total %)", f"{global_metrics.get('Missing Values (Total %)', 0):.2f}%", "HIGH" if global_metrics.get('Missing Values (Total %)', 0) > 5 else "OK"]
    ]
    
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066ff')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ])
    
    global_table = Table(global_data, colWidths=[150, 150, 100])
    global_table.setStyle(table_style)
    elements.append(global_table)
    elements.append(Spacer(1, 24))
 
    # ------------------ B. Advanced Checks ------------------
    elements.append(Paragraph("B. Advanced Checks (Correlation & Drift) ⚠️", styles['h2']))
    
    # High Correlation
    elements.append(Paragraph("Highly Correlated Numerical Pairs (r > 0.90):", styles['h3']))
    corr_data = [["Feature 1", "Feature 2", "Correlation"]]
    correlated_pairs = analysis_data['Advanced Checks'].get('Highly Correlated Pairs (> 0.90)', [])
    if correlated_pairs:
        for pair in correlated_pairs:
            corr_data.append([pair['Feature 1'], pair['Feature 2'], pair['Correlation']])
    else:
         corr_data.append(["N/A", "N/A", "None Detected"])
         
    corr_table = Table(corr_data, colWidths=[150, 150, 100])
    corr_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    elements.append(corr_table)
    elements.append(Spacer(1, 12))
    
    # Mean Drift
    elements.append(Paragraph("Data Drift Proxy (Mean Change > 10% Start vs End):", styles['h3']))
    drift_data = [["Feature", "Mean Drift"]]
    drift_metrics = analysis_data['Advanced Checks'].get('Mean Drift (Start vs End > 10% Change)', {})
    
    if drift_metrics:
        for feature, drift in drift_metrics.items():
             drift_data.append([feature, drift])
    else:
         drift_data.append(["N/A", "None Detected"])
 
    drift_table = Table(drift_data, colWidths=[200, 200])
    drift_table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ]))
    elements.append(drift_table)
    elements.append(Spacer(1, 24))
    
    # ------------------ C. Feature Breakdown Summary ------------------
    elements.append(Paragraph("C. Feature Breakdown Summary", styles['h2']))
    elements.append(Paragraph("Key metrics for columns flagged as needing attention:", styles['Normal']))
 
    col_summary_data = [["Feature", "Type", "Missing %", "Cardinality %", "Flag"]]
    
    # Prioritize columns with high missingness or a specific flag
    sorted_cols = sorted(
        analysis_data['Column Metrics'].keys(),
        key=lambda k: analysis_data['Column Metrics'][k].get('Missing Values (%)', 0),
        reverse=True
    )
    
    for col in sorted_cols:
        metrics = analysis_data['Column Metrics'][col]
        flag = metrics.get('**Status Flag**', 'OK')
        
        # Only show flagged columns for a cleaner summary
        if flag != 'OK' or metrics.get('Missing Values (%)', 0) > 10:
             col_summary_data.append([
                col,
                metrics['Data Type'],
                f"{metrics['Missing Values (%)']: .2f}%",
                f"{metrics['Cardinality (%)']: .2f}%", flag])
    
    if len(col_summary_data) > 1:
        summary_table = Table(col_summary_data, colWidths=[100, 70, 70, 90, 100])
        summary_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f0f0')),
        ]))
        elements.append(summary_table)
    else:
        elements.append(Paragraph("All features appear to be in good health (less than 10% missing, no major flags).", styles['Normal']))
    
    # Build the PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer