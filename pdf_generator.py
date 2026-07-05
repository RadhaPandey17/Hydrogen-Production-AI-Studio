"""
==========================================================
Hydrogen Production AI Studio (2026)

PDF Generator

==========================================================
"""

from io import BytesIO

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)


class PDFGenerator:

    def __init__(self):

        self.styles = getSampleStyleSheet()

    # ---------------------------------------------------

    def generate(
        self,
        prediction_result,
        ai_report
    ):

        buffer = BytesIO()

        pdf = SimpleDocTemplate(buffer)

        elements = []

        title = Paragraph(

            "<b>Hydrogen Production AI Studio</b>",

            self.styles["Title"]

        )

        elements.append(title)

        elements.append(Spacer(1, 20))

        elements.append(

            Paragraph(

                "<b>Prediction Summary</b>",

                self.styles["Heading2"]

            )

        )

        elements.append(

            Paragraph(

                f"Hydrogen Production : "
                f"{prediction_result['Hydrogen_Output']} kg/day",

                self.styles["BodyText"]

            )

        )

        elements.append(

            Paragraph(

                f"Estimated CO₂ Emission : "
                f"{prediction_result['CO2_Emission']} kg CO₂-eq/kg H₂",

                self.styles["BodyText"]

            )

        )

        elements.append(

            Paragraph(

                f"Latitude : {prediction_result['Latitude']}",

                self.styles["BodyText"]

            )

        )

        elements.append(

            Paragraph(

                f"Longitude : {prediction_result['Longitude']}",

                self.styles["BodyText"]

            )

        )

        elements.append(Spacer(1, 20))

        elements.append(

            Paragraph(

                "<b>AI Sustainability Report</b>",

                self.styles["Heading2"]

            )

        )

        for paragraph in ai_report.split("\n"):

            if paragraph.strip():

                elements.append(

                    Paragraph(

                        paragraph,

                        self.styles["BodyText"]

                    )

                )

                elements.append(Spacer(1, 6))

        pdf.build(elements)

        buffer.seek(0)

        return buffer