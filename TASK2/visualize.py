from sklearn.manifold import TSNE
import numpy as np
from pyecharts.charts import Scatter
from pyecharts import options as opts
from pyecharts.commons.utils import JsCode

# 降维并绘制前300个词的关联散点图

# TSNE降维
tsne = TSNE(n_components=2, learning_rate=100).fit_transform(embedding_weights)

x_data =[]
y_data =[]
index = 300 # 注意我们这里为了防止数据太过密集，只取前300个词来进行绘制
for i, label in enumerate(list(word2idx.keys())[:index]):
    x, y = float(tsne[i][0]), float(tsne[i][1])
    x_data.append(x)
    y_data.append((y, label))

(
    Scatter(init_opts=opts.InitOpts(width="16000px", height="10000px"))
    .add_xaxis(xaxis_data=x_data)
    .add_yaxis(
        series_name="",
        y_axis=y_data,
        symbol_size=50,
        label_opts=opts.LabelOpts(
            font_size=50,
            formatter=JsCode(
                "function(params){return params.value[2];}"
            )
        ),
    )
    .set_series_opts()
    .set_global_opts(
        xaxis_opts=opts.AxisOpts(type_="value"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
        ),
        tooltip_opts=opts.TooltipOpts(is_show=False),
    )
    .render("scatter.html")
)