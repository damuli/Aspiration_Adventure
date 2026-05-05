# 華語四聲完整檢測系統（媽→麻→馬→罵）
# 功能：
# 1. 系統依序要求使用者朗讀：媽 → 麻 → 馬 → 罵
# 2. 每次錄音後立即提供聲調回饋
# 3. 最後自動產生成績單（四張聲調曲線 + 總評）
#
# 建議執行環境：Google Colab

!pip install gradio praat-parselmouth

import gradio as gr
import parselmouth
import matplotlib.pyplot as plt
import numpy as np

# =====================
# 基本設定
# =====================

WORDS = [
    ("媽 mā", 1),
    ("麻 má", 2),
    ("馬 mǎ", 3),
    ("罵 mà", 4),
]


def extract_pitch(audio_path):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch()

    pitch_values = pitch.selected_array['frequency']
    time_points = pitch.xs()

    pitch_values[pitch_values == 0] = np.nan
    valid_pitch = pitch_values[~np.isnan(pitch_values)]

    if len(valid_pitch) < 5:
        return None

    start_pitch = valid_pitch[0]
    mid_pitch = valid_pitch[len(valid_pitch) // 2]
    end_pitch = valid_pitch[-1]
    pitch_range = np.nanmax(valid_pitch) - np.nanmin(valid_pitch)
    slope = end_pitch - start_pitch

    return {
        "time": time_points,
        "pitch": pitch_values,
        "valid": valid_pitch,
        "start": start_pitch,
        "mid": mid_pitch,
        "end": end_pitch,
        "range": pitch_range,
        "slope": slope,
    }


# =====================
# 聲調判斷
# =====================

def judge_tone(target_tone, feat):
    slope = feat["slope"]
    pitch_range = feat["range"]
    start = feat["start"]
    mid = feat["mid"]
    end = feat["end"]

    score = 60
    feedback = ""

    # 第一聲
    if target_tone == 1:
        if abs(slope) < 15 and pitch_range < 30:
            score = 95
            feedback = "第一聲表現很好：高而平穩，接近標準高平調。"
        elif slope < -20:
            score = 72
            feedback = "尾音下降較明顯，較接近第四聲，建議維持高而平。"
        else:
            score = 78
            feedback = "音高略有起伏，第一聲應更平穩。"

    # 第二聲
    elif target_tone == 2:
        if slope > 20:
            score = 92
            feedback = "第二聲表現良好：有明顯上升趨勢。"
        elif abs(slope) < 10:
            score = 75
            feedback = "上升幅度不足，較接近第一聲，尾音可再上揚。"
        elif slope < 0:
            score = 68
            feedback = "方向錯誤，出現下降趨勢，較接近第四聲。"
        else:
            score = 80
            feedback = "第二聲可再加強，建議更明顯上揚。"

    # 第三聲
    elif target_tone == 3:
        if mid < start - 15 and end >= mid:
            score = 90
            feedback = "第三聲表現不錯：具有低降特徵。"
        elif slope > 15:
            score = 72
            feedback = "較接近第二聲，第三聲需要先降再升。"
        elif abs(slope) < 10:
            score = 76
            feedback = "第三聲不夠低，建議先把音調降下去。"
        else:
            score = 79
            feedback = "第三聲特徵仍可更明顯，請加強低點。"

    # 第四聲
    elif target_tone == 4:
        if slope < -25:
            score = 94
            feedback = "第四聲表現很好：下降明顯，接近標準下降調。"
        elif abs(slope) < 10:
            score = 74
            feedback = "下降幅度不足，較接近第一聲，應快速下降。"
        elif slope > 0:
            score = 67
            feedback = "方向錯誤，較接近第二聲，第四聲應往下掉。"
        else:
            score = 80
            feedback = "第四聲可再明顯一些，建議從高音快速下降。"

    return score, feedback


# =====================
# 單張圖
# =====================

def make_single_plot(word, feat):
    plt.figure(figsize=(8, 3.8))
    plt.plot(feat["time"], feat["pitch"], linewidth=2)
    plt.title(f"{word} 聲調曲線")
    plt.xlabel("Time (s)")
    plt.ylabel("F0 (Hz)")
    plt.grid()
    plt.tight_layout()
    fig = plt.gcf()
    return fig


# =====================
# 成績單總圖
# =====================

def make_report_plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, r in enumerate(results):
        axes[i].plot(r["feat"]["time"], r["feat"]["pitch"], linewidth=2)
        axes[i].set_title(f'{r["word"]}｜{r["score"]} 分')
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("F0")
        axes[i].grid()

    plt.tight_layout()
    return fig


# =====================
# 主流程
# =====================

def analyze_all(audio1, audio2, audio3, audio4):
    audios = [audio1, audio2, audio3, audio4]
    results = []

    for (word, tone), audio in zip(WORDS, audios):
        if audio is None:
            return None, "請完成四個字的錄音後再送出。"

        feat = extract_pitch(audio)
        if feat is None:
            return None, f"{word} 錄音過短或無法分析，請重新錄音。"

        score, feedback = judge_tone(tone, feat)
        results.append({
            "word": word,
            "score": score,
            "feedback": feedback,
            "feat": feat,
        })

    avg_score = round(sum(r["score"] for r in results) / 4, 1)

    report_text = f"【華語四聲檢測成績單】\n\n總平均：{avg_score} 分\n\n"

    for r in results:
        report_text += f"{r['word']}：{r['score']} 分\n"
        report_text += f"評語：{r['feedback']}\n\n"

    if avg_score >= 90:
        report_text += "總評：你的四聲掌握非常穩定，已接近自然華語發音。"
    elif avg_score >= 80:
        report_text += "總評：整體表現良好，部分聲調仍可再加強。"
    else:
        report_text += "總評：建議持續練習四聲對比，特別注意聲調方向。"

    report_fig = make_report_plot(results)

    return report_fig, report_text


# =====================
# Gradio UI
# =====================

with gr.Blocks() as demo:
    gr.Markdown("# 華語四聲完整檢測系統")
    gr.Markdown("## 請依序朗讀：媽 → 麻 → 馬 → 罵，完成後系統將自動產生成績單")

    with gr.Row():
        audio1 = gr.Audio(type="filepath", label="第一題：請唸『媽 mā』")
        audio2 = gr.Audio(type="filepath", label="第二題：請唸『麻 má』")

    with gr.Row():
        audio3 = gr.Audio(type="filepath", label="第三題：請唸『馬 mǎ』")
        audio4 = gr.Audio(type="filepath", label="第四題：請唸『罵 mà』")

    submit_btn = gr.Button("產生成績單", variant="primary")

    output_plot = gr.Plot(label="四聲檢測總圖")
    output_text = gr.Textbox(label="完整評語與成績單", lines=18)

    submit_btn.click(
        fn=analyze_all,
        inputs=[audio1, audio2, audio3, audio4],
        outputs=[output_plot, output_text]
    )


demo.launch()
