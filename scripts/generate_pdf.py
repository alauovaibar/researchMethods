"""
PDF Generation Script for Methodology Passport Card.
Uses ReportLab with Windows system Unicode font (Arial) to generate a clean PDF document.
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def generate_methodology_card_pdf(output_pdf: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_pdf)), exist_ok=True)

    # Register Arial font for Cyrillic support
    font_path = "C:\\Windows\\Fonts\\arial.ttf"
    font_bold_path = "C:\\Windows\\Fonts\\arialbd.ttf"
    font_italic_path = "C:\\Windows\\Fonts\\ariali.ttf"

    font_name = "Helvetica"
    font_bold = "Helvetica-Bold"
    font_italic = "Helvetica-Oblique"

    if os.path.exists(font_path):
        pdfmetrics.registerFont(TTFont("Arial", font_path))
        font_name = "Arial"
        if os.path.exists(font_bold_path):
            pdfmetrics.registerFont(TTFont("Arial-Bold", font_bold_path))
            font_bold = "Arial-Bold"
        else:
            font_bold = "Arial"
        if os.path.exists(font_italic_path):
            pdfmetrics.registerFont(TTFont("Arial-Italic", font_italic_path))
            font_italic = "Arial-Italic"
        else:
            font_italic = "Arial"

    doc = SimpleDocTemplate(
        output_pdf,
        pagesize=A4,
        leftMargin=28,
        rightMargin=28,
        topMargin=26,
        bottomMargin=26,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CardTitle",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#1A365D"),
        alignment=1, # Center
    )

    subtitle_style = ParagraphStyle(
        "CardSubTitle",
        parent=styles["Normal"],
        fontName=font_italic,
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
    )

    h1_style = ParagraphStyle(
        "CardH1",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#2B6CB0"),
        spaceBefore=7,
        spaceAfter=3,
    )

    body_style = ParagraphStyle(
        "CardBody",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor("#2D3748"),
    )

    body_bold = ParagraphStyle(
        "CardBodyBold",
        parent=body_style,
        fontName=font_bold,
    )

    cell_style = ParagraphStyle(
        "CardCell",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#2D3748"),
    )

    cell_header = ParagraphStyle(
        "CardCellHeader",
        parent=styles["Normal"],
        fontName=font_bold,
        fontSize=8,
        leading=10,
        textColor=colors.white,
    )

    story = []

    # Title
    story.append(Paragraph("КАРТОЧКА МЕТОДОЛОГИИ ИССЛЕДОВАНИЯ", title_style))
    story.append(Paragraph("Methodology Passport Card | Course: Research Methods | Task 2", subtitle_style))
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2B6CB0"), spaceAfter=10))

    # Section 1: Project Information
    story.append(Paragraph("1. Информация о проекте (Project Information)", h1_style))
    project_info = [
        [
            Paragraph("<b>Название работы:</b>", body_style),
            Paragraph("Влияние конфигурации извлечения (top-k и размер фрагмента) на уровень галлюцинаций в вопросно-ответных системах с генерацией на основе поиска (RAG)", body_style),
        ],
        [
            Paragraph("<b>English Title:</b>", body_style),
            Paragraph("Effect of Retrieval Configuration (top-k and chunk size) on the Hallucination Rate of Retrieval-Augmented Question Answering with an Open-Weight LLM", body_style),
        ],
        [
            Paragraph("<b>Автор / Пара:</b>", body_style),
            Paragraph("Айбар Алауов (Aibar Alauov) — aibar.alauov@narxoz.com", body_style),
        ],
        [
            Paragraph("<b>GitHub Репозиторий:</b>", body_style),
            Paragraph("<font color='#2B6CB0'><u>https://github.com/alauovaibar/researchMethods</u></font>", body_style),
        ],
    ]
    t_proj = Table(project_info, colWidths=[110, 425])
    t_proj.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 1),
    ]))
    story.append(t_proj)
    story.append(Spacer(1, 4))

    # Section 2: Research Type
    story.append(Paragraph("2. Тип исследования (Research Type)", h1_style))
    story.append(Paragraph("<b>Тип:</b> Количественное экспериментальное исследование (Quantitative Empirical Benchmark Study).", body_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>Обоснование выбора:</b> Исследование направлено на выявление эмпирических причинно-следственных зависимостей и многофакторную оптимизацию гиперпараметров извлечения информации (top-k, chunk_size, позиция контекста) на объективно измеримые количественные метрики качества ответов LLM (Hallucination Rate, Faithfulness, Token F1, задержка инференса). Проверка гипотез выполняется с использованием дисперсионного анализа (ANOVA) и post-hoc теста Тьюки (α = 0.05).", body_style))
    story.append(Spacer(1, 4))

    # Section 3: Hypotheses
    story.append(Paragraph("3. Формулировка гипотез (Hypotheses Formulation)", h1_style))
    story.append(Paragraph("<b>Нулевая гипотеза (H0):</b> Средний уровень галлюцинаций (Hallucination Rate) генерируемых ответов не зависит от параметров извлечения top-k (1, 3, 5, 10, 20) и размера чанка (128, 256, 512, 1024 токенов): μ_1 = μ_3 = μ_5 = μ_10 = μ_20 (α = 0.05).", body_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>Альтернативная гипотеза (H1):</b> Существует статистически значимая U-образная зависимость между глубиной извлечения и уровнем галлюцинаций: при малых k (k=1) уровень галлюцинаций повышен из-за нехватки доказательств; в диапазоне k ∈ [3, 5] достигается минимум галлюцинаций; при k ≥ 10 уровень галлюцинаций возрастает вследствие отвлекающего шума пассажей (distractor noise) и рассеивания внимания декодера.", body_style))
    story.append(Spacer(1, 4))

    # Section 4: Variables Matrix Table
    story.append(Paragraph("4. Матрица переменных (Variables Matrix)", h1_style))
    vars_data = [
        [Paragraph("Категория", cell_header), Paragraph("Переменная", cell_header), Paragraph("Тип / Значения", cell_header), Paragraph("Роль в эксперименте", cell_header)],
        [Paragraph("<b>Независимая (IV1)</b>", cell_style), Paragraph("top_k", cell_style), Paragraph("1, 3, 5, 10, 20", cell_style), Paragraph("Глубина извлечения пассажей из векторного индекса", cell_style)],
        [Paragraph("<b>Независимая (IV2)</b>", cell_style), Paragraph("chunk_size", cell_style), Paragraph("128, 256, 512, 1024", cell_style), Paragraph("Гранулярность разбиения документов корпуса", cell_style)],
        [Paragraph("<b>Независимая (IV3)</b>", cell_style), Paragraph("gold_position", cell_style), Paragraph("first, middle, last, auto", cell_style), Paragraph("Позиция эталона для проверки 'Lost in the Middle'", cell_style)],
        [Paragraph("<b>Зависимая (DV1)</b>", cell_style), Paragraph("Hallucination Rate", cell_style), Paragraph("0.0 - 1.0 (доля)", cell_style), Paragraph("<b>Первичная метрика:</b> доля неподтвержденных фактов", cell_style)],
        [Paragraph("<b>Зависимая (DV2)</b>", cell_style), Paragraph("Faithfulness Score", cell_style), Paragraph("0.0 - 1.0 (RAGAs)", cell_style), Paragraph("Фактическая верность сгенерированного ответа контексту", cell_style)],
        [Paragraph("<b>Зависимая (DV3)</b>", cell_style), Paragraph("Token F1 / EM", cell_style), Paragraph("0.0 - 1.0", cell_style), Paragraph("Согласованность с эталонным экспертным ответом", cell_style)],
        [Paragraph("<b>Зависимая (DV4)</b>", cell_style), Paragraph("Latency (мс)", cell_style), Paragraph("> 0 мс", cell_style), Paragraph("Суммарное время поиска и генерации", cell_style)],
        [Paragraph("<b>Контролируемая (CV1)</b>", cell_style), Paragraph("LLM Модель", cell_style), Paragraph("Llama-3-8B-Instruct", cell_style), Paragraph("Фиксированная открытая генеративная модель", cell_style)],
        [Paragraph("<b>Контролируемая (CV2)</b>", cell_style), Paragraph("Параметры декодирования", cell_style), Paragraph("temp=0.0, seed=42", cell_style), Paragraph("Строгая детерминированная жадная генерация", cell_style)],
        [Paragraph("<b>Контролируемая (CV3)</b>", cell_style), Paragraph("Модель эмбеддингов", cell_style), Paragraph("Dense Cosine (dim=384)", cell_style), Paragraph("Единое координатное пространство сходства", cell_style)],
        [Paragraph("<b>Контролируемая (CV4)</b>", cell_style), Paragraph("Промпт-шаблон", cell_style), Paragraph("Strict Factual RAG", cell_style), Paragraph("Единый системный промпт без изменения формулировок", cell_style)],
    ]
    t_vars = Table(vars_data, colWidths=[90, 100, 115, 230])
    t_vars.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2B6CB0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E0")),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#F7FAFC"), colors.white]),
    ]))
    story.append(t_vars)
    story.append(Spacer(1, 4))

    # Section 5: Metrics & Baseline
    story.append(Paragraph("5. Выбранные метрики и Baseline (Metrics & Baseline)", h1_style))
    story.append(Paragraph("<b>Первичная метрика (Primary Metric):</b> Hallucination Rate (доля неподтвержденных утверждений в ответе) и Faithfulness Score = 1.0 - HR. Значение 0.0 означает 100% обоснованность текста.", body_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>Пороговые метрики (Guardrail Metrics):</b> P95 Latency ≤ 2500 мс; Контекст ≤ 8192 токенов; Error Rate = 0.0%; RAM ≤ 4096 МБ.", body_style))
    story.append(Spacer(1, 2))
    story.append(Paragraph("<b>Контрольная группа / Baseline:</b> Baseline 0 (No-RAG, k=0) — генерация без контекста (базовая склонность к галлюцинациям); Baseline 1 (Standard Vanilla RAG) — индустриальный стандарт по умолчанию: top_k=3, chunk_size=256, auto positioning.", body_style))

    doc.build(story)
    print(f"Successfully generated PDF: {output_pdf}")


if __name__ == "__main__":
    generate_methodology_card_pdf("docs/METHODOLOGY_CARD.pdf")
    generate_methodology_card_pdf("METHODOLOGY_CARD.pdf")
