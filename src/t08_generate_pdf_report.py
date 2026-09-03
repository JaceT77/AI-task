import structlog
from fpdf import FPDF
from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.getLogger(__name__)

chart_source: str = "./scorer_results/candidate_december.png"
pdf_target: str = "./src/data/Freight_Rate_ML_Report.pdf"


def generate_pdf() -> None:
    log.info(event="initializing PDF generation")

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font(family="Arial", style="B", size=16)
    pdf.cell(w=0, h=10, txt="Freight Rate ML Prediction Report", ln=True, align="C")

    # Executive Summary
    pdf.set_font(family="Arial", size=12)
    pdf.ln(h=10)
    summary_text: str = (
        "This report summarizes the Gradient Boosting model performance and the "
        "December seasonal analysis. The validation gap between training and testing "
        "was a healthy 3.97%, and distance accounted for over 90% of the pricing variance."
    )
    pdf.multi_cell(w=0, h=8, txt=summary_text)

    # Embedded Chart
    pdf.ln(h=10)
    pdf.set_font(family="Arial", style="B", size=14)
    pdf.cell(w=0, h=10, txt="December 2025 Rate Forecast", ln=True)

    try:
        pdf.image(name=chart_source, x=10, y=pdf.get_y(), w=190)
        log.info(event="chart image successfully embedded", source=chart_source)
    except Exception as exc:
        log.error(event="failed to embed chart image", error=str(exc))

    pdf.output(name=pdf_target)
    log.info(event="PDF report successfully saved", target=pdf_target)


if __name__ == "__main__":
    generate_pdf()
