import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether

# Create reports directory
os.makedirs("reports", exist_ok=True)
os.makedirs("reports/figures", exist_ok=True)


def generate_pdf_report(record_name: str, fs: float, sig_raw: np.ndarray, sig_processed: np.ndarray, 
                        peaks: np.ndarray, rr_metrics: dict, fft_freqs: np.ndarray, fft_mags: np.ndarray, 
                        predicted_label: str, confidence: float, model_name: str, explanation_text: str,
                        quality_status: str, quality_details: str):
    """
    Generates a professional PDF report containing the complete ECG signal processing,
    feature analysis and machine-learning result. Saves the output to 'reports/'.
    """
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_filename = f"reports/ecg_report_{record_name}_{timestamp_str}.pdf"
    
    # --- Generate matplotlib figures to embed ---
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # 1. DSP Signal comparison
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 3.5), sharex=True)
    t = np.arange(len(sig_raw)) / fs
    # Display 10 seconds of signal
    disp_limit = min(len(sig_raw), int(10 * fs))
    ax1.plot(t[:disp_limit], sig_raw[:disp_limit], color='#E74C3C', alpha=0.85, label='Raw ECG')
    ax1.set_title("ECG Preprocessing Stage 1: Raw Artifact-Affected Signal", fontsize=9, fontweight='bold', color='#2C3E50')
    ax1.set_ylabel("Amplitude (mV)", fontsize=8)
    ax1.legend(loc='upper right', prop={'size': 7})
    
    ax2.plot(t[:disp_limit], sig_processed[:disp_limit], color='#2ECC71', alpha=0.9, label='Filtered ECG')
    ax2.set_title("ECG Preprocessing Stage 2: Clean Baseline-Removed and Filtered Signal", fontsize=9, fontweight='bold', color='#2C3E50')
    ax2.set_xlabel("Time (seconds)", fontsize=8)
    ax2.set_ylabel("Amplitude (mV)", fontsize=8)
    ax2.legend(loc='upper right', prop={'size': 7})
    plt.tight_layout()
    
    dsp_fig_path = f"reports/figures/dsp_compare_{timestamp_str}.png"
    plt.savefig(dsp_fig_path, dpi=200)
    plt.close()
    
    # 2. Peak detection & RR intervals
    fig2, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(t[:disp_limit], sig_processed[:disp_limit], color='#3498DB', alpha=0.9, label='Clean ECG')
    # Filter peaks to the 10-second display range
    disp_peaks = peaks[peaks < disp_limit]
    ax.scatter(disp_peaks / fs, sig_processed[disp_peaks], color='#E74C3C', s=25, zorder=5, label='Detected R-peaks')
    ax.set_title("Time-Domain Analysis: Pan-Tompkins Peak Alignment", fontsize=9, fontweight='bold', color='#2C3E50')
    ax.set_xlabel("Time (seconds)", fontsize=8)
    ax.set_ylabel("Amplitude (mV)", fontsize=8)
    ax.legend(loc='upper right', prop={'size': 7})
    plt.tight_layout()
    
    peaks_fig_path = f"reports/figures/peaks_{timestamp_str}.png"
    plt.savefig(peaks_fig_path, dpi=200)
    plt.close()
    
    # 3. FFT Analysis
    fig3, ax = plt.subplots(figsize=(8, 2.5))
    # We display up to 60 Hz for clarity
    disp_freq_mask = fft_freqs <= 60.0
    ax.plot(fft_freqs[disp_freq_mask], fft_mags[disp_freq_mask], color='#9B59B6', linewidth=1.2)
    ax.set_title("Frequency-Domain Analysis: FFT Magnitude Spectrum", fontsize=9, fontweight='bold', color='#2C3E50')
    ax.set_xlabel("Frequency (Hz)", fontsize=8)
    ax.set_ylabel("Magnitude", fontsize=8)
    plt.tight_layout()
    
    fft_fig_path = f"reports/figures/fft_{timestamp_str}.png"
    plt.savefig(fft_fig_path, dpi=200)
    plt.close()

    # --- Setup ReportLab Document ---
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor('#1E3D59') # Deep Navy
    c_secondary = colors.HexColor('#17B890') # Tealy Green
    c_accent = colors.HexColor('#E74C3C') # Red
    
    # Custom Paragraph Styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.white,
        alignment=1, # Center
        spaceAfter=15
    )
    
    section_h1 = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=6,
        borderPadding=2
    )
    
    cell_label_style = ParagraphStyle(
        'CellLabel',
        fontName='Helvetica-Bold',
        fontSize=9,
        textColor=colors.HexColor('#2C3E50')
    )
    
    cell_val_style = ParagraphStyle(
        'CellVal',
        fontName='Helvetica',
        fontSize=9,
        textColor=colors.HexColor('#34495E')
    )
    
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        textColor=colors.HexColor('#7F8C8D'),
        alignment=0,
        spaceBefore=8
    )

    story = []
    
    # 1. Header Banner
    banner_data = [[Paragraph("INTELLIGENT ECG ANALYSIS REPORT", title_style)]]
    banner_table = Table(banner_data, colWidths=[532])
    banner_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_primary),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 15),
    ]))
    story.append(banner_table)
    story.append(Spacer(1, 10))
    
    # 2. Key Metadata & Quality Info Table
    duration_sec = len(sig_raw) / fs
    analysis_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Format signal quality text colour
    quality_color = '#2ECC71' # Green
    if quality_status == 'POOR':
         quality_color = '#E74C3C' # Red
    elif quality_status == 'ACCEPTABLE':
         quality_color = '#F39C12' # Amber
         
    quality_html = f"<b><font color='{quality_color}'>{quality_status}</font></b>"
    
    metadata_data = [
        [Paragraph("Record ID:", cell_label_style), Paragraph(record_name, cell_val_style),
         Paragraph("Analysis Timestamp:", cell_label_style), Paragraph(analysis_time, cell_val_style)],
        
        [Paragraph("Sampling Rate:", cell_label_style), Paragraph(f"{fs} Hz", cell_val_style),
         Paragraph("Signal Quality Status:", cell_label_style), Paragraph(quality_html, cell_val_style)],
        
        [Paragraph("Signal Duration:", cell_label_style), Paragraph(f"{duration_sec:.1f} Sec ({len(sig_raw)} samples)", cell_val_style),
         Paragraph("Quality Details:", cell_label_style), Paragraph(quality_details, cell_val_style)]
    ]
    
    metadata_table = Table(metadata_data, colWidths=[95, 171, 120, 146])
    metadata_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#BDC3C7')),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#ECF0F1')),
        ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#ECF0F1')),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(metadata_table)
    story.append(Spacer(1, 10))
    
    # 3. Machine Learning Classification Results Dashboard
    story.append(Paragraph("Machine Learning Rhythm Classification", section_h1))
    
    # Calibrate display formatting for class
    class_full_names = {
        'N': 'Normal Sinus Beat (N)',
        'V': 'Premature Ventricular Contraction (V / PVC)',
        'S': 'Supraventricular Ectopic Beat (S / APC)',
        'F': 'Fusion Beat (F)',
    }
    predicted_full = class_full_names.get(predicted_label, predicted_label)
    
    ml_data = [
        [Paragraph("Classification Outcome:", cell_label_style), Paragraph(f"<b><font size='10' color='#1E3D59'>{predicted_full}</font></b>", cell_val_style)],
        [Paragraph("ML Model Used:", cell_label_style), Paragraph(model_name, cell_val_style)],
        [Paragraph("Model Confidence Level:", cell_label_style), Paragraph(f"<b>{confidence:.1%}</b>", cell_val_style)]
    ]
    
    ml_table = Table(ml_data, colWidths=[140, 392])
    ml_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#BDC3C7')),
        ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#ECF0F1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(ml_table)
    story.append(Spacer(1, 8))
    
    # 4. Clinical Explainability Section
    clean_exp_text = explanation_text.replace("**", "").replace("###", "").strip()
    # Format markdown bullet points slightly for pdf Paragraph flowable
    clean_exp_text = clean_exp_text.replace("\n", "<br/>")
    
    exp_style = ParagraphStyle(
        'ExpertSystemExplanation',
        fontName='Helvetica',
        fontSize=8.5,
        textColor=colors.HexColor('#2C3E50'),
        leading=11
    )
    
    exp_box_data = [[Paragraph(clean_exp_text, exp_style)]]
    exp_box_table = Table(exp_box_data, colWidths=[532])
    exp_box_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8F9F9')),
        ('BOX', (0,0), (-1,-1), 1.0, colors.HexColor('#1E3D59')),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(exp_box_table)
    story.append(Spacer(1, 10))
    
    # 5. DSP Preprocessing Waveforms
    story.append(Paragraph("Digital Signal Processing Pipeline Details", section_h1))
    story.append(Image(dsp_fig_path, width=500, height=218))
    story.append(Spacer(1, 12))
    
    # 6. Time Domain and Frequency Domain Waveforms (Force to keep together on next page if needed)
    analysis_flowables = [
        Paragraph("Signal Analysis: Time & Frequency Domains", section_h1),
        Spacer(1, 4),
        
        # HRV & Waveform features table
        Table([
            [Paragraph("Total R-peaks Detected", cell_label_style), Paragraph(str(rr_metrics['num_beats']), cell_val_style),
             Paragraph("Mean Heart Rate", cell_label_style), Paragraph(f"{rr_metrics['mean_hr']:.1f} BPM", cell_val_style)],
            [Paragraph("Mean RR Interval", cell_label_style), Paragraph(f"{rr_metrics['mean_rr']:.1f} ms", cell_val_style),
             Paragraph("SDRR (RR Variance)", cell_label_style), Paragraph(f"{rr_metrics['std_rr_sdrr']:.1f} ms", cell_val_style)],
            [Paragraph("RMSSD (HRV index)", cell_label_style), Paragraph(f"{rr_metrics['rmssd']:.1f} ms", cell_val_style),
             Paragraph("Pacemaker Pacing", cell_label_style), Paragraph("Bradycardia (&lt;50)" if rr_metrics['mean_hr'] < 50 else ("Tachycardia (&gt;100)" if rr_metrics['mean_hr'] > 100 else "Normal Sinus Pacing"), cell_val_style)]
        ], colWidths=[130, 136, 130, 136], style=TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#BDC3C7')),
            ('BACKGROUND', (0,0), (0,-1), colors.HexColor('#F8F9F9')),
            ('BACKGROUND', (2,0), (2,-1), colors.HexColor('#F8F9F9')),
            ('PADDING', (0,0), (-1,-1), 4),
        ])),
        Spacer(1, 8),
        
        # Image 2 (R-peaks alignment)
        Image(peaks_fig_path, width=500, height=156),
        Spacer(1, 8),
        
        # Image 3 (FFT frequency spectrum)
        Image(fft_fig_path, width=500, height=156),
    ]
    story.append(KeepTogether(analysis_flowables))
    story.append(Spacer(1, 10))
    
    # 7. Disclaimer block at bottom of document
    disclaimer_text = (
        "<b>Academic & Research Prototype Disclaimer:</b> This software and generated report are intended solely "
        "for academic, research, and educational demonstrations of biomedical digital signal processing and machine-learning "
        "classifications. It has <b>NOT</b> been audited, certified, or approved by any medical device authority (such as the CDSCO, FDA, or CE) "
        "and is <b>NOT</b> a medical diagnostic system. Under no circumstances should these findings be used to replace direct "
        "professional medical diagnosis, hospital-grade ECG equipment, or clinical decision-making. The classification accuracy "
        "and parameters are subject to algorithms and modeling limits."
    )
    
    disclaimer_box = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[532])
    disclaimer_box.setStyle(TableStyle([
        ('LINELEFT', (0,0), (0,-1), 3.0, c_accent),
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#FDEDEC')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(disclaimer_box)
    
    # Build Document
    doc.build(story)
    
    # Cleanup temporary matplotlib images from disk
    try:
        os.remove(dsp_fig_path)
        os.remove(peaks_fig_path)
        os.remove(fft_fig_path)
    except Exception as e:
        print(f"Warning: could not delete temporary images: {e}")
        
    return pdf_filename
