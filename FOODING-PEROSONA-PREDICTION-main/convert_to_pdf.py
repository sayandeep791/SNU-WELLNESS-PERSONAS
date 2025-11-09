from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from pathlib import Path
import re

md_file = Path('SNU_Wellness_Personas_Report.md')
pdf_file = Path('SNU_Wellness_Personas_Report.pdf')

with open(md_file, 'r', encoding='utf-8') as f:
    md_content = f.read()

doc = SimpleDocTemplate(str(pdf_file), pagesize=A4,
                        rightMargin=0.75*inch, leftMargin=0.75*inch,
                        topMargin=0.75*inch, bottomMargin=0.75*inch)

styles = getSampleStyleSheet()

styles.add(ParagraphStyle(name='CustomTitle',
                          parent=styles['Heading1'],
                          fontSize=24,
                          textColor=colors.HexColor('#1e3a8a'),
                          spaceAfter=20,
                          alignment=TA_CENTER,
                          fontName='Helvetica-Bold'))

styles.add(ParagraphStyle(name='Heading1Custom',
                          parent=styles['Heading1'],
                          fontSize=18,
                          textColor=colors.HexColor('#1e3a8a'),
                          spaceAfter=12,
                          spaceBefore=12,
                          fontName='Helvetica-Bold'))

styles.add(ParagraphStyle(name='Heading2Custom',
                          parent=styles['Heading2'],
                          fontSize=14,
                          textColor=colors.HexColor('#2563eb'),
                          spaceAfter=10,
                          spaceBefore=10,
                          fontName='Helvetica-Bold'))

styles.add(ParagraphStyle(name='Heading3Custom',
                          parent=styles['Heading3'],
                          fontSize=12,
                          textColor=colors.HexColor('#3b82f6'),
                          spaceAfter=8,
                          spaceBefore=8,
                          fontName='Helvetica-Bold'))

styles.add(ParagraphStyle(name='BodyTextCustom',
                          parent=styles['BodyText'],
                          fontSize=10,
                          alignment=TA_JUSTIFY,
                          spaceAfter=6))

story = []

lines = md_content.split('\n')
i = 0
first_h1 = True

while i < len(lines):
    line = lines[i].strip()
    
    if line.startswith('# '):
        text = line[2:].strip()
        if first_h1:
            story.append(Paragraph(text, styles['CustomTitle']))
            first_h1 = False
        else:
            story.append(PageBreak())
            story.append(Paragraph(text, styles['Heading1Custom']))
        story.append(Spacer(1, 0.2*inch))
    
    elif line.startswith('## '):
        text = line[3:].strip()
        story.append(Paragraph(text, styles['Heading2Custom']))
        story.append(Spacer(1, 0.1*inch))
    
    elif line.startswith('### '):
        text = line[4:].strip()
        story.append(Paragraph(text, styles['Heading3Custom']))
        story.append(Spacer(1, 0.1*inch))
    
    elif line.startswith('---'):
        story.append(Spacer(1, 0.15*inch))
    
    elif line.startswith('**') and line.endswith('**'):
        text = line[2:-2]
        story.append(Paragraph(f'<b>{text}</b>', styles['BodyTextCustom']))
        story.append(Spacer(1, 0.05*inch))
    
    elif line.startswith('- ') or line.startswith('* '):
        bullet_text = line[2:].strip()
        bullet_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', bullet_text)
        story.append(Paragraph(f'• {bullet_text}', styles['BodyTextCustom']))
    
    elif re.match(r'^\d+\.', line):
        num_text = re.sub(r'^\d+\.\s*', '', line)
        num_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', num_text)
        story.append(Paragraph(num_text, styles['BodyTextCustom']))
    
    elif line and not line.startswith('#'):
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', line)
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        story.append(Paragraph(text, styles['BodyTextCustom']))
        story.append(Spacer(1, 0.05*inch))
    
    elif not line:
        story.append(Spacer(1, 0.1*inch))
    
    i += 1

doc.build(story)

print(f"✅ PDF generated successfully: {pdf_file.absolute()}")
