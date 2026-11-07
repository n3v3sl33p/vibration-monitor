"""Local single-process laboratory UI; session arrays live only in session_state."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from acquisition import run_session
from generator import SignalConfig


def main():
    st.set_page_config(page_title="ЛР2 · Вибрация M1", layout="wide")
    st.title("ЛР2 · Виртуальный электродвигатель M1")
    st.caption("Условный датчик вибрации, амплитуда в усл. ед. Синтетические отклонения — не диагноз двигателя.")
    st.info("Исследователь редактирует параметры; оператор запускает сеанс и просматривает результат. "
            "Это сценарные роли, не авторизация.")
    st.caption("A/F по умолчанию — контрольный пример ЛР1, НЕ назначенный числовой вариант. "
               "Время t = n/fs; конечный сеанс рассчитывается блоками без ожидания реального времени.")
    with st.form("settings"):
        st.subheader("Исследователь · параметры сеанса")
        amplitudes = st.text_input("Амплитуды A (через запятую)", "1, 0.5, 0.25", key="amplitudes")
        frequencies = st.text_input("Частоты F, Гц (через запятую)", "0.05, 0.1, 0.4", key="frequencies")
        left, right = st.columns(2)
        with left:
            fs = st.number_input("Частота выборки fs, Гц", value=20.0, key="fs")
            duration = st.number_input("Длительность, с", value=60.0, key="duration")
            noise = st.number_input("Стандартное отклонение шума", value=0.1, key="noise_std")
            seed = st.number_input("Seed", value=42, step=1, key="seed")
            period = st.number_input("Период EMA N", value=3, step=1, key="ema_period")
            st.caption("Период 3 выбран экспериментально на контрольном сеансе; настраивается")
        with right:
            scenario = st.selectbox("Сценарий", ("normal", "impulse", "amplitude", "frequency"), key="scenario")
            event_start = st.number_input("Начало события, с", value=20.0, key="event_start")
            event_end = st.number_input("Конец события (не включается), с", value=30.0, key="event_end")
            amplitude_factor = st.number_input("Множитель амплитуды", value=2.0, key="amplitude_factor")
            frequency_factor = st.number_input("Множитель частоты первой гармоники", value=2.0, key="frequency_factor")
            impulse_width = st.number_input("Ширина единственного импульса, с", value=0.1, key="impulse_width")
            impulse_height = st.number_input("Высота импульса, усл. ед.", value=2.0, key="impulse_height")
        st.caption("АЦП: 12 бит, диапазон [−4, 4], ближайший уровень. Блок: 128 отсчётов. "
                   "Не более 1 000 000 отсчётов в сеансе. Шум добавляется дискретно; "
                   "аналоговый антиалиасинговый фильтр не моделируется. "
                   "Импульс начинается в начале интервала; частотное событие сохраняет непрерывность фазы.")
        submitted = st.form_submit_button("Оператор · запустить сеанс")

    if submitted:
        # Remove stale results BEFORE parsing/validation, not just after a successful run.
        st.session_state['result'] = None
        st.session_state.pop('result_config', None)
        st.session_state.pop('result_period', None)
        st.session_state['error'] = None
        try:
            config = SignalConfig(
                amplitudes=tuple(float(item.strip()) for item in amplitudes.split(',')),
                frequencies=tuple(float(item.strip()) for item in frequencies.split(',')),
                fs=fs, duration=duration, noise_std=noise, seed=seed, scenario=scenario,
                event_start=event_start, event_end=event_end, amplitude_factor=amplitude_factor,
                frequency_factor=frequency_factor, impulse_width=impulse_width, impulse_height=impulse_height)
            st.session_state['result'] = run_session(config, ema_period=period)
            st.session_state['result_config'] = config
            st.session_state['result_period'] = period
        except ValueError as error:
            st.session_state['error'] = f"Некорректные параметры: {error}"

    if st.session_state.get('error'):
        st.error(st.session_state['error'])

    result = st.session_state.get('result')
    if result is None:
        st.info("Настройте параметры и запустите сеанс.")
        return
    config = st.session_state['result_config']
    period = st.session_state['result_period']
    st.subheader("Оператор · результат принятого сеанса")
    st.caption(f"Сценарий: {config.scenario}; seed={config.seed}; fs={config.fs:g} Гц; "
               f"длительность={config.duration:g} с; N={period}; α={2 / (period + 1):g}.")
    if result.saturation_count:
        st.warning(f"Насыщение АЦП: {result.saturation_count} отсчётов вне диапазона [−4, 4]; "
                   "они ограничены границами диапазона.")
    st.metric("Принято отсчётов", len(result.t))
    chart = go.Figure()
    for label, values in (("Исходный дискретный", result.raw),
                          ("Квантованный", result.quantized), ("EMA", result.filtered)):
        chart.add_trace(go.Scatter(x=result.t, y=values, name=label, mode="lines"))
    chart.update_layout(xaxis_title="Время, с", yaxis_title="Вибрация, усл. ед.",
                        title="Принятый сигнал и предварительная обработка", height=420)
    st.plotly_chart(chart, width="stretch", key="signals")
    st.caption("Эталон генератора — НЕ обнаружение. Маска показывает заданное синтетическое событие, "
               "а не результат диагностики или ML; у импульса отмечена только его поддержка.")
    truth = go.Figure(go.Scatter(x=result.t, y=result.mask.astype(int), mode="lines",
                                line_shape="hv", fill="tozeroy", name="Эталон генератора"))
    truth.update_layout(xaxis_title="Время, с", yaxis_title="Эталон (0/1)",
                        yaxis=dict(tickvals=[0, 1], range=[-0.05, 1.05]), height=200)
    st.plotly_chart(truth, width="stretch", key="reference_mask")
    st.dataframe(pd.DataFrame({"Время, с": result.t, "Исходный": result.raw,
                               "Квантованный": result.quantized, "EMA": result.filtered,
                               "Эталон генератора": result.mask}), width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
