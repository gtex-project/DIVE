from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import os
import uuid
import glob
import matplotlib.pyplot as plt

from plot_figures import plot_hydrogen_material_trend, plot_hydrogen_density_distribution, plot_material_type_distribution, plot_typical_material_element_analyses  # 你也可以按需添加其他绘图函数

app = FastAPI()

from fastapi.responses import HTMLResponse

@app.get("/privacy", response_class=HTMLResponse)
async def privacy_policy():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Privacy Policy</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; padding: 20px; max-width: 800px; margin: auto; }
            h1, h2 { color: #2c3e50; }
        </style>
    </head>
    <body>
        <h1>Privacy Policy</h1>
        <p>This GPT-based service uses the <code>/plot</code> endpoint to generate visualizations of hydrogen storage materials based on user input.</p>

        <h2>1. What data do we collect?</h2>
        <p>We only collect parameters required to generate plots (such as material type, year range, or element lists). No personal or identifying information is collected.</p>

        <h2>2. How is data used?</h2>
        <p>Your input is used solely to produce visual outputs. The data is processed temporarily in memory and not stored or logged persistently.</p>

        <h2>3. Do we share data?</h2>
        <p>No data is shared with third parties. All data processing occurs securely on our server.</p>

        <h2>4. Data retention</h2>
        <p>Images generated are temporarily stored and automatically deleted after a certain limit is reached. Input data is discarded immediately after use.</p>

        <h2>5. Contact</h2>
        <p>If you have questions or requests regarding privacy, please contact: <a href="mailto:di.zhang.a8@tohoku.ac.jp">di.zhang.a8@tohoku.ac.jp</a></p>
    </body>
    </html>
    """

# 图片缓存设置
STATIC_DIR = "static/images"
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/images", StaticFiles(directory=STATIC_DIR), name="images")

MAX_IMAGES = 100
DELETE_COUNT = 50

def cleanup_old_images():
    image_files = glob.glob(os.path.join(STATIC_DIR, "*.png"))
    if len(image_files) > MAX_IMAGES:
        image_files.sort(key=os.path.getctime)  # 最旧的在前
        for old_file in image_files[:DELETE_COUNT]:
            try:
                os.remove(old_file)
            except Exception as e:
                print(f"Failed to delete {old_file}: {e}")

# 请求数据模型
class PlotRequest(BaseModel):
    figure_type: str
    year_range: Optional[List[int]] = [1972, 2025]
    material_type: Optional[List[str]] = [
        "Interstitial Hydride", "Complex Hydride", "Multi-component Hydride",
        "Porous Material", "Ionic Hydride", "Superhydride", "Others"
    ]
    number_of_interested_elements: Optional[str] = "5"
    interested_performance: Optional[List[str]] = []
    elements_in_typical_material: Optional[List[str]] = []
    color_list: Optional[List[str]] = ['#1f77b4','#adc6ea', '#ff8800', '#fbb78f', '#98df8a', '#d62728', '#8c564b']


@app.post("/plot")
async def plot(request: PlotRequest):
    database_path = "/home/dizhang/langgraph/PaperReadingAgent/Data_Analyses/1st_published_dighyd/Total_Data_V250728.csv"
    cleanup_old_images()
    filename = f"{uuid.uuid4().hex}.png"
    image_path = os.path.join(STATIC_DIR, filename)

    # 根据 figure_type 选择绘图逻辑
    print(request)
    if request.figure_type == "publication_trend":
        plot_hydrogen_material_trend(
            csv_path=database_path,
            image_path=image_path,
            year_range=request.year_range,
            material_type=request.material_type,
            color_list=request.color_list
        )

    elif request.figure_type == "material_type_based_trend":
        plot_hydrogen_density_distribution(
            csv_path=database_path,
            image_path=image_path,
            year_range=request.year_range,
            material_type=request.material_type,
            number_of_interested_elements=request.number_of_interested_elements,
            interested_performance=request.interested_performance,
            color_list=request.color_list
        )

    elif request.figure_type == "material_type_ratio":
        plot_material_type_distribution(
            csv_path=database_path,
            image_path=image_path,
            year_range=request.year_range,
            material_type=request.material_type,
            color_list=request.color_list
        )

    elif request.figure_type == "typical_material_analyses":
        plot_typical_material_element_analyses(
            csv_path=database_path,
            image_path=image_path,
            material_type_targets=request.material_type,
            element_targets=request.elements_in_typical_material,
            interested_col=request.interested_performance[0] if request.interested_performance else 'Dehydrogenation temperature_processed',
            top_n=int(request.number_of_interested_elements) if request.number_of_interested_elements else 5,
            color_list=request.color_list
        )

    else:
        return JSONResponse(status_code=400, content={"error": "Unsupported figure_type"})

    image_url = f"https://plot.dighyd.org/images/{filename}"
    print({"markdown": f"![Plot]({image_url})"})
    return JSONResponse(content={
            "markdown": f"![Plot]({image_url})"
        })
