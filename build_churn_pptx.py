# -*- coding: utf-8 -*-
"""Minimal white deck: target, dataset, features."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

HW = Path(__file__).resolve().parent
DATA = HW / "data"
OUT = HW / "Churn_pokaz.pptx"
OUT_ALT = HW / "Churn_pokaz_lag5.pptx"
CHART_OUT = DATA / "rolling_backtest_chart.png"

CHART_INK = "#1A1A1A"
CHART_MUTED = "#6B6B6B"
CHART_LINE = "#E6E6E6"
CHART_ACCENT = "#4D9FFF"
CHART_WARN = "#B85C38"

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x6B, 0x6B, 0x6B)
LINE = RGBColor(0xE6, 0xE6, 0xE6)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

W = Inches(13.333)
H = Inches(7.5)
ML = Inches(0.72)
MR = Inches(0.72)


def _run(p, text, size=18, bold=False, color=INK):
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = "Calibri"
    return r


def tf_box(slide, l, t, w, h):
    box = slide.shapes.add_textbox(l, t, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    return tf


def write(tf, lines):
    """lines: list of dicts text/size/bold/color/after/align"""
    first = True
    for item in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = item.get("align", PP_ALIGN.LEFT)
        p.space_after = Pt(item.get("after", 6))
        p.space_before = Pt(item.get("before", 0))
        _run(p, item["text"], item.get("size", 16), item.get("bold", False), item.get("color", INK))


def hline(slide, y):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, ML, y, W - ML - MR, Emu(12700))
    sh.fill.solid()
    sh.fill.fore_color.rgb = LINE
    sh.line.fill.background()


def new_slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = WHITE
    bg.line.fill.background()
    spTree = s.shapes._spTree
    sp = bg._element
    spTree.remove(sp)
    spTree.insert(2, sp)
    return s


def add_table(slide, left, top, width, height, rows, font_size=13):
    n_rows, n_cols = len(rows), len(rows[0])
    ts = slide.shapes.add_table(n_rows, n_cols, left, top, width, height)
    table = ts.table
    widths = [int(width * w / sum(_col_weights(rows))) for w in _col_weights(rows)]
    for i, w in enumerate(widths):
        table.columns[i].width = w
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.cell(r, c)
            cell.text = ""
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT
            _run(p, str(val), font_size, r == 0, MUTED if r == 0 else INK)
            tcPr = cell._tc.get_or_add_tcPr()
            for child in list(tcPr):
                if any(x in child.tag for x in ("solidFill", "gradFill")):
                    tcPr.remove(child)
            for border in ("lnL", "lnR", "lnT", "lnB"):
                el = etree.SubElement(tcPr, qn(f"a:{border}"))
                el.set("w", "6350")
                sf = etree.SubElement(el, qn("a:solidFill"))
                srgb = etree.SubElement(sf, qn("a:srgbClr"))
                srgb.set("val", "E6E6E6")
    return table


def _col_weights(rows):
    n = len(rows[0])
    if n == 2:
        return [1.15, 1.85]
    if n == 3:
        return [1.4, 1.1, 1.5]
    if n == 4:
        return [1.4, 1.6, 0.7, 0.7]
    if n == 5:
        return [1.1] * 5
    if n == 7:
        return [1.7, 0.9, 0.7, 0.6, 0.6, 1.0, 1.0]
    return [1] * n


def add_picture(slide, path, left, top, width):
    slide.shapes.add_picture(str(path), Inches(left), Inches(top), width=Inches(width))


def build_backtest_chart(csv_path: Path = DATA / "rolling_backtest.csv") -> Path:
    df = pd.read_csv(csv_path, parse_dates=["test_snap"])
    df = df.sort_values("test_snap")
    df["label"] = df["test_snap"].dt.strftime("%Y-%m")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": CHART_LINE,
            "axes.labelcolor": CHART_INK,
            "xtick.color": CHART_MUTED,
            "ytick.color": CHART_MUTED,
            "text.color": CHART_INK,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.grid": True,
            "grid.color": CHART_LINE,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax1 = plt.subplots(figsize=(7.2, 3.8))
    x = range(len(df))

    ax1.plot(x, df["auc"], marker="o", linewidth=2.2, color=CHART_ACCENT, label="AUC")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(df["label"], rotation=0)
    ax1.set_ylim(0.78, 0.93)
    ax1.set_ylabel("AUC", color=CHART_ACCENT)
    ax1.tick_params(axis="y", labelcolor=CHART_ACCENT)

    ax_ap = ax1.twinx()
    ax_ap.plot(x, df["ap"], marker="s", linewidth=2.0, color=CHART_WARN, label="AP")
    ax_ap.set_ylim(0.22, 0.55)
    ax_ap.set_ylabel("AP", color=CHART_WARN)
    ax_ap.tick_params(axis="y", labelcolor=CHART_WARN)
    ax_ap.spines.top.set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax_ap.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

    ax2 = ax1.twinx()
    ax2.spines.right.set_position(("outward", 52))
    ax2.bar(
        x,
        df["test_rate"] * 100,
        width=0.35,
        alpha=0.15,
        color=CHART_MUTED,
        label="test churn %",
    )
    ax2.set_ylim(5.0, 8.5)
    ax2.set_ylabel("test churn, %", color=CHART_MUTED)
    ax2.spines.top.set_visible(False)
    ax2.spines.right.set_color(CHART_LINE)
    ax2.tick_params(axis="y", colors=CHART_MUTED)

    dip_pos = int(df["auc"].values.argmin())
    ax1.annotate(
        "просадка\nперед тарифом",
        xy=(dip_pos, df.iloc[dip_pos]["auc"]),
        xytext=(dip_pos, df.iloc[dip_pos]["auc"] - 0.08),
        ha="center",
        fontsize=9,
        color=CHART_MUTED,
        arrowprops={"arrowstyle": "->", "color": CHART_MUTED, "lw": 1.0},
    )

    ax1.set_title("Out-of-time backtest: train t → test t+1", loc="left", fontsize=11, pad=12)
    fig.tight_layout()
    CHART_OUT.parent.mkdir(exist_ok=True)
    fig.savefig(CHART_OUT, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return CHART_OUT


def heading(s, kicker, title):
    tf = tf_box(s, ML, Inches(0.42), W - ML - MR, Inches(1.15))
    write(
        tf,
        [
            {"text": kicker, "size": 11, "color": MUTED, "after": 6},
            {"text": title, "size": 28, "bold": True, "after": 0},
        ],
    )
    hline(s, Inches(1.22))


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    cw = W - ML - MR

    # 1 cover
    s = new_slide(prs)
    tf = tf_box(s, ML, Inches(2.15), cw, Inches(4))
    write(
        tf,
        [
            {"text": "ОТТОК", "size": 12, "color": MUTED, "after": 14},
            {"text": "Таргет · датасет · сценарий · выгода", "size": 34, "bold": True, "after": 16},
            {"text": "Churn prevention, не uplift-модель. academy_db / churn_* · до 31.12.2025", "size": 16, "color": MUTED, "after": 0},
        ],
    )

    # 2 target
    s = new_slide(prs)
    heading(s, "ТАРГЕТ", "90 дней без approved после snapshot")
    tf = tf_box(s, ML, Inches(1.4), cw, Inches(1.5))
    write(
        tf,
        [
            {
                "text": "target = 1, если клиент жив на snapshot и в следующие 90 дней нет ни одной approved-операции. Иначе 0.",
                "size": 18,
                "bold": True,
                "after": 8,
            },
            {
                "text": "Не берём customers_360.is_churned (там тишина 31 день).",
                "size": 15,
                "color": MUTED,
                "after": 0,
            },
        ],
    )
    add_table(
        s,
        ML,
        Inches(3.15),
        cw,
        Inches(2.4),
        [
            ["", "Порог", "Лаг", "Train", "Test"],
            ["Правило", "90 дней", "6 дней", "2025-06-30", "2025-09-25"],
            ["Смысл", "окно метки", "старт snap+6", "06.07–04.10", "01.10–30.12"],
        ],
        14,
    )
    tf = tf_box(s, ML, Inches(5.8), cw, Inches(1.1))
    write(
        tf,
        [
            {
                "text": "Лаг 6 после snap: между скорингом и стартом окна метки. Фичи < snap, target с snap+6. Churn, не uplift.",
                "size": 15,
                "color": MUTED,
            }
        ],
    )

    # 3 why 90
    s = new_slide(prs)
    heading(s, "ТАРГЕТ", "Почему 90, а не 30")
    add_table(
        s,
        ML,
        Inches(1.5),
        cw,
        Inches(3.5),
        [
            ["", "Число", "Вывод"],
            ["p50 / p90 / p95 / p99 gap", "3 / 12 / 17 / 37 дн.", "обычный ритм, не уход"],
            ["return после 28 дней тишины", "83%", "порог 30 ловит живых"],
            ["return после 91 дня тишины", "~26%", "уже уход"],
            ["пример", "gap 121, txn 24.12.2025", "длинная пауза ≠ ушёл"],
        ],
        15,
    )
    tf = tf_box(s, ML, Inches(5.3), cw, Inches(1.6))
    write(
        tf,
        [
            {
                "text": "Gap — ритм между операциями. Return rate — вернулся ли после тишины. Таргет по второму.",
                "size": 16,
                "after": 0,
            }
        ],
    )

    # 4 horizon sensitivity
    s = new_slide(prs)
    heading(s, "ТАРГЕТ", "Проверка устойчивости порога")
    add_table(
        s,
        ML,
        Inches(1.5),
        cw,
        Inches(3.2),
        [
            ["Горизонт", "N", "target=1", "доля", "Как читать"],
            ["60 дней", "12 367", "1 112", "8.99%", "рано, ловит больше временной тишины"],
            ["90 дней", "12 367", "875", "7.08%", "баланс между паузой и реальным уходом"],
            ["120 дней", "12 367", "814", "6.58%", "поздно для реакции бизнеса"],
        ],
        15,
    )
    tf = tf_box(s, ML, Inches(5.2), cw, Inches(1.6))
    write(
        tf,
        [
            {
                "text": "Сравнили соседние горизонты на одном и том же train-срезе. 90 дней не максимум и не минимум, а рабочий компромисс.",
                "size": 16,
                "after": 8,
            },
            {
                "text": "60 дней даёт больше churn, потому что часть клиентов ещё вернулась бы позже. 120 дней спокойнее, но слишком поздно для удержания.",
                "size": 14,
                "color": MUTED,
                "after": 0,
            },
        ],
    )

    # 5 population
    s = new_slide(prs)
    heading(s, "ТАРГЕТ", "Кого размечаем")
    add_table(
        s,
        ML,
        Inches(1.5),
        cw,
        Inches(4.0),
        [
            ["В выборке", "Не в выборке"],
            ["tenure ≥ 60 дней", "новички"],
            ["approved за 90д до snap", "уже молчит > 90 дней"],
            ["snap + lag + 90 ≤ 31.12.2025", "окно не закрыто"],
            ["фичи только дата < snap", "360, NPS ноя–дек, будущее"],
        ],
        15,
    )
    tf = tf_box(s, ML, Inches(5.75), cw, Inches(1.1))
    write(
        tf,
        [
            {
                "text": "label_start = snap + 6. Test snap 25.09, чтобы окно закрылось к 31.12.",
                "size": 15,
                "color": MUTED,
            }
        ],
    )

    # 6 dataset
    s = new_slide(prs)
    heading(s, "ДАТАСЕТ", "Два среза, не random split")
    add_table(
        s,
        ML,
        Inches(1.5),
        cw,
        Inches(2.5),
        [
            ["", "Snapshot", "label+6", "N", "target=1", "доля"],
            ["train", "30.06.2025", "06.07", "12 265", "789", "6.43%"],
            ["test", "25.09.2025", "01.10", "12 876", "956", "7.42%"],
        ],
        15,
    )
    tf = tf_box(s, ML, Inches(4.3), cw, Inches(2.5))
    write(
        tf,
        [
            {
                "text": "Строка = клиент на дату. Фичи до snap, target с snap+6 на 90д. txn_next_h в модель не идёт.",
                "size": 16,
                "after": 10,
            },
            {
                "text": "Источник: customers, transactions (approved), cards, tickets, subscriptions, charges, tariff_changes, app_events, product_events.",
                "size": 14,
                "color": MUTED,
                "after": 0,
            },
        ],
    )

    # 7 rolling backtest
    backtest_csv = DATA / "rolling_backtest.csv"
    chart_path = build_backtest_chart(backtest_csv) if backtest_csv.exists() else None
    s = new_slide(prs)
    heading(s, "МОДЕЛЬ", "Исторический backtest по кварталам")
    add_table(
        s,
        ML,
        Inches(1.45),
        Inches(5.6),
        Inches(3.7),
        [
            ["train → test", "test rate", "AUC", "AP"],
            ["2024-06 → 2024-09", "5.53%", "0.861", "0.317"],
            ["2024-09 → 2024-12", "5.56%", "0.852", "0.308"],
            ["2024-12 → 2025-03", "5.30%", "0.868", "0.391"],
            ["2025-03 → 2025-06", "6.43%", "0.796", "0.261"],
            ["2025-06 → 2025-09", "7.42%", "0.901", "0.515"],
        ],
        13,
    )
    if chart_path is not None:
        add_picture(s, chart_path, 6.55, 1.42, 6.35)
    tf = tf_box(s, ML, Inches(5.35), Inches(5.8), Inches(1.35))
    write(
        tf,
        [
            {
                "text": "Учим на квартале t, проверяем на t+1. График: AUC/AP и фактический churn на test.",
                "size": 14,
                "after": 6,
            },
            {
                "text": "Просадка на 2025-06 совпадает с изменением поведения перед тарифом.",
                "size": 13,
                "color": MUTED,
                "after": 0,
            },
        ],
    )

    # 8 monthly
    s = new_slide(prs)
    heading(s, "ДАТАСЕТ", "Отток вырос после тарифа")
    add_table(
        s,
        ML,
        Inches(1.45),
        Inches(7.3),
        Inches(4.6),
        [
            ["snap", "живых", "ушли 90д", "%"],
            ["2024-07", "9 565", "614", "6.42"],
            ["2024-08", "9 781", "600", "6.13"],
            ["2025-06", "12 048", "593", "4.92"],
            ["2025-07", "12 278", "785", "6.39"],
            ["2025-08", "12 560", "907", "7.22"],
            ["2025-09", "12 777", "986", "7.72"],
        ],
        14,
    )
    tf = tf_box(s, Inches(8.25), Inches(1.5), Inches(4.4), Inches(4.8))
    write(
        tf,
        [
            {"text": "1 июля 2025", "size": 14, "color": MUTED, "after": 6},
            {"text": "8 139 → 23 793 сум", "size": 22, "bold": True, "after": 8},
            {"text": "tariff_revision_2025", "size": 14, "color": MUTED, "after": 14},
            {"text": "Лето 2024: 6.42 → 6.13. Не сезон.", "size": 15, "after": 0},
        ],
    )

    # 9 features
    s = new_slide(prs)
    heading(s, "ФИЧИ", "As-of snapshot, относительно самого клиента")
    add_table(
        s,
        ML,
        Inches(1.45),
        cw,
        Inches(5.5),
        [
            ["блок", "фичи", "ушли vs остались"],
            ["ритм", "rhythm_break / pause_is_record", "6.4 vs 1.9 · 38% vs 6%"],
            ["переезд", "mono_fade / bank_switch_score", "21% vs 9% лестница"],
            ["быт", "days_without_grocery / pos", "95 vs 41 · 59 vs 21"],
            ["", "abandoned_favorite / empty_weeks", "76% vs 43% · 6.6 vs 3.9"],
            ["тариф", "fee_pain / unbundling", "закрыл продукт 14% vs 0.8%"],
            ["мы", "retention_failed", "11% vs 0.1%"],
            ["соседи", "weirder_than_peers", "1.75σ vs 0.29σ"],
        ],
        14,
    )

    # 10 drop
    s = new_slide(prs)
    heading(s, "ФИЧИ", "Проверили и не берём")
    add_table(
        s,
        ML,
        Inches(1.5),
        cw,
        Inches(4.6),
        [
            ["гипотеза", "почему нет"],
            ["digital_ghost", "логины есть и у живых"],
            ["missed_payroll / expiry_soon", "почти всегда 0"],
            ["deposit_drain / loan_winddown", "у оставшихся чаще"],
            ["NPS", "только ноя–дек 2025"],
            ["is_churned из 360", "чужой таргет + leakage"],
        ],
        15,
    )

    # 10b feature importance
    s = new_slide(prs)
    heading(s, "МОДЕЛЬ", "Feature importance (permutation, test)")
    add_table(
        s,
        ML,
        Inches(1.45),
        Inches(7.4),
        Inches(4.8),
        [
            ["#", "фича", "Δ AP", "смысл"],
            ["1", "retention_failed", "+0.085", "уже писали — не помогло"],
            ["2", "txn_cnt_90", "+0.069", "активность за квартал"],
            ["3", "app_events_30", "+0.049", "жизнь в приложении"],
            ["4", "recency_days", "+0.045", "текущая тишина"],
            ["5", "txn_cnt_30", "+0.039", "недавняя активность"],
            ["6", "is_payroll", "+0.030", "зарплатный клиент"],
            ["7", "cards_dead", "+0.026", "закрытые / мёртвые карты"],
            ["8", "empty_weeks", "+0.024", "дыры в недельном ритме"],
            ["9", "sub_price", "+0.021", "цена активной подписки"],
            ["10", "bank_switch_score", "+0.017", "лестница: карта не основная"],
        ],
        13,
    )
    tf = tf_box(s, Inches(9.1), Inches(1.5), Inches(3.7), Inches(5.0))
    write(
        tf,
        [
            {"text": "Как считали", "size": 14, "color": MUTED, "after": 6},
            {
                "text": "Перемешали колонку на test → насколько упал AP. Чем больше Δ AP, тем важнее фича.",
                "size": 14,
                "after": 14,
            },
            {"text": "Не путать с lift", "size": 14, "color": MUTED, "after": 6},
            {
                "text": "«Ушли vs остались» — гипотеза до модели. Importance — чем модель реально пользуется.",
                "size": 14,
                "after": 14,
            },
            {"text": "Вывод", "size": 14, "color": MUTED, "after": 6},
            {
                "text": "В топе RFM и свои: retention_failed, empty_weeks, bank_switch. pause_is_record тоже значим (+0.014).",
                "size": 14,
                "after": 0,
            },
        ],
    )

    # 11 scenario
    s = new_slide(prs)
    heading(s, "СЦЕНАРИЙ", "С кем взаимодействовали и что было после")
    add_table(
        s,
        ML,
        Inches(1.45),
        cw,
        Inches(3.3),
        [
            ["Группа", "Кто", "N", "вернулись 90д"],
            ["Control", "тишина 30–90, без контакта", "18 894", "71%"],
            ["Treatment", "писали / звонили", "1 516", "4%"],
            ["30–45, control", "короткая тишина, без контакта", "11 065", "82%"],
            ["30–45, treatment", "короткая тишина, с контактом", "894", "3%"],
        ],
        14,
    )
    tf = tf_box(s, ML, Inches(5.0), cw, Inches(1.8))
    write(
        tf,
        [
            {
                "text": "Кампании FEE_WAIVER / WINBACK / CALL_RETENTION били в уже тихих. Control часто возвращается сам — это пауза.",
                "size": 15,
                "after": 6,
            },
            {
                "text": "Это селекция, не A/B uplift. Дальше — prevention на ещё живых, не повтор winback по «трупам».",
                "size": 15,
                "color": MUTED,
                "after": 0,
            },
        ],
    )

    # 12 value
    s = new_slide(prs)
    heading(s, "ВЫГОДА", "Не средний LTV с рынка")
    add_table(
        s,
        ML,
        Inches(1.45),
        cw,
        Inches(3.0),
        [
            ["Что", "Откуда", "Число"],
            ["комиссия карты", "churn_tariff_changes", "8 139 → 23 793 сум"],
            ["подписка, если активна", "churn_subscriptions", "медиана ~29 000 сум/мес"],
            ["формула", "наши таблицы", "value_90d = 3 × (комиссия + подписка)"],
            ["потери на test", "956 ушедших × их value", "~74.5 млн сум / квартал"],
        ],
        14,
    )
    tf = tf_box(s, ML, Inches(4.8), cw, Inches(2.0))
    write(
        tf,
        [
            {
                "text": "×3, потому что горизонт таргета 90 дней. Если удержали — банк получает квартал маржи. Дальше не рисуем.",
                "size": 15,
                "after": 6,
            },
            {
                "text": "Средняя value на test ~78.8 тыс. сум. Без «среднего по рынку».",
                "size": 14,
                "color": MUTED,
                "after": 0,
            },
        ],
    )

    # 13 metrics + threshold
    s = new_slide(prs)
    heading(s, "МЕТРИКИ", "Порог = бюджет, не 0.5")
    add_table(
        s,
        ML,
        Inches(1.45),
        Inches(6.0),
        Inches(3.5),
        [
            ["", "Число", "Смысл"],
            ["AUC / AP (OOT)", "0.886 / 0.422", "ранжирование; класс редкий"],
            ["порог", "top-K по бюджету", "не p = 0.5"],
            ["top-200 по p×value", "77 уходов · 20.0 млн", "сначала риск × маржа"],
            ["top-500 по p×value", "149 уходов · 30.1 млн", "расширяем слоты"],
            ["top-1000 по p×value", "256 уходов · 41.0 млн", "capture растёт"],
        ],
        13,
    )
    tf = tf_box(s, Inches(7.7), Inches(1.5), Inches(5.0), Inches(5.0))
    write(
        tf,
        [
            {"text": "Что не берём", "size": 14, "color": MUTED, "after": 6},
            {"text": "Accuracy и порог 0.5 — класс ~7%, бюджет не на всех.", "size": 14, "after": 12},
            {"text": "Как режем список", "size": 14, "color": MUTED, "after": 6},
            {"text": "Сортируем по p(churn) × value_retain_90d. Звоним ещё живым с высоким риском и маржой.", "size": 14, "after": 12},
            {"text": "Это не uplift-модель", "size": 14, "color": MUTED, "after": 6},
            {"text": "Uplift — отдельный вопрос A/B. У нас churn-скоринг + экономика слота.", "size": 14, "after": 0},
        ],
    )

    try:
        prs.save(OUT)
        print("saved", OUT)
    except PermissionError:
        prs.save(OUT_ALT)
        print("saved", OUT_ALT, "(close Churn_pokaz.pptx and replace)")


if __name__ == "__main__":
    build()
