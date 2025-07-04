import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from geopy.distance import geodesic
import folium
from streamlit_folium import st_folium
import google.generativeai as genai

# --- パスワード認証機能 ---
def check_password():
    if st.session_state.get("password_correct", False):
        return True
    st.set_page_config(layout="centered", page_title="Login - Race Data Dashboard")
    st.title("🏍️ Pro-Spec Race Data Dashboard")
    st.write("---")
    password = st.text_input("パスワードを入力してください", type="password", key="password_input")
    if st.button("ログイン", key="login_button"):
        if hasattr(st.secrets, "passwords") and password in st.secrets.passwords.values():
            st.session_state.password_correct = True
            st.rerun()
        else:
            st.error("😕 パスワードが正しくありません。")
    st.info("このアプリケーションを閲覧するにはパスワードが必要です。")
    return False

if not check_password():
    st.stop()

# --- 0. Streamlit アプリケーションの基本設定 ---
st.set_page_config(layout="wide", page_title="🏍️ Pro-Spec Race Data Dashboard", page_icon="🏍️")
st.title("🏍️ Pro-Spec Race Data Dashboard")
with st.expander("ℹ️ 各データ項目の意味を表示する"):
    st.markdown("""
    - **Speed:** マシンの速度 (km/h)。CSVの`Speed_kmh`列を使用。
    - **Brake Pressure & G:** ブレーキ操作とそれに伴う減速G。`F-Brake`はフロント、`R-Brake`はリアのブレーキ圧。`Decel-G`は前後G。
    - **Pitch & Lean Angle:** `Pitch (GYPIAN)`はマシンの前後傾斜（ジャイロセンサーによる）、`Lean`は左右の傾き（バンク角）。車両姿勢の基本。
    - **A/F Ratio & Throttle:** `AFR`または`LAF1`は空燃比（エンジンの燃焼状態）、`Throttle`はスロットル開度(%)。
    - **Gear & RPM:** `RPM`はエンジン回転数、`Gear`はギアポジション。
    - **Torque Reduction:** `TRQPCTRD`は電子制御によるトルクカット率(%)。トラクションコントロール(TC)の介入度を示す。
    - **Wheel Slip:** `SLIP`はタイヤのスリップ率(%)または前後輪の速度差(km/h)。トラクション状態を示す。
    - **TC, AFR, Lean (on Map):** マップ上のマーカー。`TC(34)`はTC介入、`Anti-jerk(81)`は衝撃緩和制御、`Ideal AFR`は理想空燃比、`Deep Lean`は深いバンク角の区間を示します。
    """)
st.markdown("---")

# --- 1. データの読み込み ---
with st.sidebar:
    st.header("⚙️ 設定")
    uploaded_file = st.file_uploader("比較データを含むCSVファイルをアップロードしてください", type=["csv"])
    label1 = st.text_input("データ1のラベル", value="データセット1")
    label2 = st.text_input("データ2のラベル", value="データセット2")

def load_and_split_data(file):
    if file is None: return None
    try:
        df = pd.read_csv(file)
        if 'Time' not in df.columns: st.error("CSVファイルに 'Time' 列が見つかりません。"); return None
        time_diffs = df['Time'].diff()
        split_index = time_diffs[time_diffs < 0].index.min()
        if pd.isna(split_index):
            st.warning("ファイル内にデータの区切りが見つかりませんでした。1つの走行データとして扱います。")
            df['Lap'] = 1
            return df
        st.success(f"{split_index}行目でデータの区切りを検出しました。")
        df1 = df.iloc[:split_index].copy(); df1['Lap'] = 1
        df2 = df.iloc[split_index:].copy(); df2['Lap'] = 2
        return pd.concat([df1, df2], ignore_index=True)
    except Exception as e:
        st.error(f"ファイルの読み込みまたは分割処理中にエラー: {e}"); return None

df_input = load_and_split_data(uploaded_file)

if df_input is None:
    st.info("サイドバーから比較データを含むCSVファイルを1つアップロードしてください。")
    st.stop()
    
with st.sidebar.expander("ℹ️ デバッグ情報"):
    st.write("分割・結合後データ (先頭5行):"); st.dataframe(df_input.head())
    st.write("存在する列:", df_input.columns.tolist())

# --- 2 & 3. データ処理とラップ計算 (キャッシュされた関数) ---
@st.cache_data
def process_race_data(input_df):
    df_processed = input_df.copy(); required_cols = ['Time', 'Lap']
    if not all(col in df_processed.columns for col in required_cols): st.error("エラー: CSVに 'Time' と 'Lap' 列が必要です。'Lap'列は自動で生成されます。"); st.stop()
    has_gps = 'Latitude' in df_processed.columns and 'Longitude' in df_processed.columns
    processed_laps = []
    for lap_num, group in df_processed.groupby('Lap'):
        lap_group = group.copy()
        if has_gps:
            try:
                lap_group['Timestamp'] = pd.to_timedelta(lap_group['Time'], unit='s'); lap_group.set_index('Timestamp', inplace=True)
                resample_rate = '100L'; agg_rules = {col: 'mean' for col in lap_group.columns if np.issubdtype(lap_group[col].dtype, np.number)}
                agg_rules.update({'Lap': 'first', 'Gear_Position': 'last', 'TC_Intervention': 'max'})
                valid_agg_rules = {k: v for k, v in agg_rules.items() if k in lap_group.columns}
                resampled_group = lap_group.resample(resample_rate).agg(valid_agg_rules).dropna(subset=['Lap'])
                resampled_group['Time'] = resampled_group.index.total_seconds()
                lap_group = resampled_group.reset_index(drop=True)
            except Exception as e:
                st.warning(f"Lap {lap_num} のリサンプリング処理中にエラー: {e}。処理をスキップします。")
                lap_group = lap_group.reset_index(drop=True)
            lap_group['Prev_Latitude'] = lap_group['Latitude'].shift(1); lap_group['Prev_Longitude'] = lap_group['Longitude'].shift(1); lap_group.fillna(0, inplace=True)
            def calculate_distance_row(row):
                if row['Prev_Latitude'] == 0: return 0.0
                try: return geodesic((row['Prev_Latitude'], row['Prev_Longitude']), (row['Latitude'], row['Longitude'])).meters
                except ValueError: return 0.0
            lap_group['Segment_Distance'] = lap_group.apply(calculate_distance_row, axis=1)
            lap_group['Distance'] = lap_group['Segment_Distance'].cumsum()
            lap_group.drop(['Prev_Latitude', 'Prev_Longitude', 'Segment_Distance'], axis=1, inplace=True, errors='ignore')
        else:
            lap_group['Distance'] = lap_group['Time'] - lap_group['Time'].iloc[0]
        if 'GYXAC' not in lap_group.columns:
            if 'Speed_kmh' in lap_group.columns:
                speed_ms = lap_group['Speed_kmh'] / 3.6
                delta_time = lap_group['Time'].diff().fillna(1e-9)
                delta_speed = speed_ms.diff().fillna(0)
                acceleration = delta_speed / delta_time
                lap_group['GYXAC'] = acceleration / 9.80665
            else: lap_group['GYXAC'] = 0
        processed_laps.append(lap_group)
    if not processed_laps: return pd.DataFrame()
    return pd.concat(processed_laps, ignore_index=True)

@st.cache_data
def get_comparison_laps(_processed_df, _label1, _label2):
    laps_to_plot_data = []; lap_labels_list = []
    lap1_data = _processed_df[_processed_df['Lap'] == 1].copy()
    if not lap1_data.empty:
        if 'Distance' in lap1_data.columns: lap1_data['Lap_Distance'] = lap1_data['Distance']
        laps_to_plot_data.append(lap1_data); lap_labels_list.append(_label1)
    lap2_data = _processed_df[_processed_df['Lap'] == 2].copy()
    if not lap2_data.empty:
        if 'Distance' in lap2_data.columns: lap2_data['Lap_Distance'] = lap2_data['Distance']
        laps_to_plot_data.append(lap2_data); lap_labels_list.append(_label2)
    if len(laps_to_plot_data) < 2: return [], []
    return laps_to_plot_data, lap_labels_list

# --- 4. Plotlyグラフ作成 (キャッシュされた関数) ---
@st.cache_data
def create_plotly_chart(_laps_to_plot, _lap_labels, _df_columns):
    plot_configs = [
        {'title': 'Speed', 'cols': ['Speed_kmh'], 'secondary_y': False, 'yaxis_title': 'Speed (km/h)'},
        {'title': 'Brake Pressure & G', 'cols': ['PBKF', 'PBKR', 'GYXAC'], 'secondary_y': True, 'yaxis_title': 'Brake Press.', 'yaxis2_title': 'Decel (G)'},
        {'title': 'Pitch & Lean Angle', 'cols': ['GYPIAN', 'Lean_Angle'], 'secondary_y': True, 'yaxis_title': 'Pitch (deg)', 'yaxis2_title': 'Lean (deg)'},
        {'title': 'A/F Ratio & Throttle', 'cols': [('AFR', 'LAF1'), 'Throttle_pct'], 'secondary_y': True, 'yaxis_title': 'A/F Ratio', 'yaxis2_title': 'Throttle (%)'},
        {'title': 'Gear & RPM', 'cols': ['RPM', 'Gear_Position'], 'secondary_y': True, 'yaxis_title': 'RPM', 'yaxis2_title': 'Gear'},
        {'title': 'Torque Reduction', 'cols': ['TRQPCTRD'], 'secondary_y': False, 'yaxis_title': 'Torque Cut (%)'},
        {'title': 'Wheel Slip', 'cols': ['SLIP'], 'secondary_y': False, 'yaxis_title': 'Slip'}
    ]
    active_plots = []
    for p in plot_configs:
        cols_exist = True
        for col_or_tuple in p['cols']:
            if isinstance(col_or_tuple, tuple):
                if not any(c in _df_columns for c in col_or_tuple): cols_exist = False; break
            else:
                if col_or_tuple not in _df_columns: cols_exist = False; break
        if cols_exist: active_plots.append(p)
    if not active_plots: return None, 0
    num_plots = len(active_plots); specs = [[{"secondary_y": p['secondary_y']}] for p in active_plots]; titles = [p['title'] for p in active_plots]
    fig = make_subplots(rows=num_plots, cols=1, shared_xaxes=True, subplot_titles=titles, vertical_spacing=0.03, specs=specs)
    lap_colors_plotly = ['#FF0000', '#0000FF']
    for i, plot_info in enumerate(active_plots):
        row_num = i + 1
        for j, lap_data in enumerate(_laps_to_plot):
            current_label = _lap_labels[j]; current_color = lap_colors_plotly[j % len(lap_colors_plotly)]; show_legend = (i == 0)
            if plot_info['title'] == 'Brake Pressure & G':
                fig.add_trace(go.Scatter(x=lap_data['Lap_Distance'], y=lap_data['PBKF'], mode='lines', name=f"F-Brake ({current_label})", legendgroup=current_label, showlegend=show_legend, line=dict(color=current_color)), row=row_num, col=1, secondary_y=False)
                fig.add_trace(go.Scatter(x=lap_data['Lap_Distance'], y=lap_data['PBKR'], mode='lines', name=f"R-Brake ({current_label})", legendgroup=current_label, showlegend=False, line=dict(color=current_color, dash='dot')), row=row_num, col=1, secondary_y=False)
                fig.add_trace(go.Scatter(x=lap_data['Lap_Distance'], y=-lap_data['GYXAC'], mode='lines', name=f"Decel-G ({current_label})", legendgroup=current_label, showlegend=False, line=dict(color=current_color, width=1, dash='dashdot')), row=row_num, col=1, secondary_y=True)
            elif plot_info['title'] == 'Pitch & Lean Angle':
                fig.add_trace(go.Scatter(x=lap_data['Lap_Distance'], y=lap_data['GYPIAN'], mode='lines', name=f"Pitch ({current_label})", legendgroup=current_label, showlegend=show_legend, line=dict(color=current_color)), row=row_num, col=1, secondary_y=False)
                fig.add_trace(go.Scatter(x=lap_data['Lap_Distance'], y=lap_data['Lean_Angle'], mode='lines', name=f"Lean ({current_label})", legendgroup=current_label, showlegend=False, line=dict(color=current_color, dash='dot')), row=row_num, col=1, secondary_y=True)
            elif plot_info['title'] == 'A/F Ratio & Throttle':
                af_col_name = 'LAF1' if 'LAF1' in lap_data.columns else 'AFR'
                y_data = lap_data[af_col_name]
                if af_col_name == 'LAF1': y_data = y_data * 14.7
                fig.add_trace(go.Scatter(x=lap_data['Lap_Distance'], y=y_data, mode='lines', name=current_label, legendgroup=current_label, showlegend=show_legend, line=dict(color=current_color)), row=row_num, col=1, secondary_y=False)
                fig.add_trace(go.Scatter(x=lap_data['Lap_Distance'], y=lap_data['Throttle_pct'], mode='lines', name="Throttle (%)", legendgroup=current_label, showlegend=False, line=dict(color=current_color, dash='dot', width=1)), row=row_num, col=1, secondary_y=True)
            elif plot_info['title'] == 'Gear & RPM':
                fig.add_trace(go.Scatter(x=lap_data['Lap_Distance'], y=lap_data['RPM'], mode='lines', name=current_label, legendgroup=current_label, showlegend=show_legend, line=dict(color=current_color)), row=row_num, col=1, secondary_y=False)
                fig.add_trace(go.Scatter(x=lap_data['Lap_Distance'], y=lap_data['Gear_Position'], mode='lines', name="Gear", legendgroup=current_label, showlegend=False, line=dict(color=current_color, dash='dot', width=1), line_shape='hv'), row=row_num, col=1, secondary_y=True)
            else:
                y_data = lap_data[plot_info['cols'][0]]
                if 'transform' in plot_info: y_data = plot_info['transform'](y_data)
                fig.add_trace(go.Scatter(x=lap_data['Lap_Distance'], y=y_data, mode='lines', name=current_label, legendgroup=current_label, showlegend=show_legend, line=dict(color=current_color)), row=row_num, col=1)
    fig.update_layout(height=250 * num_plots, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), hovermode="x unified", autosize=True, margin=dict(l=40, r=40, b=40, t=60, pad=4), plot_bgcolor='rgba(240, 242, 246, 0.95)')
    for i, plot_info in enumerate(active_plots):
        row_num = i + 1; fig.update_yaxes(title_text=plot_info['yaxis_title'], row=row_num, col=1, secondary_y=False)
        if plot_info['secondary_y']: fig.update_yaxes(title_text=plot_info['yaxis2_title'], row=row_num, col=1, secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Distance (m) or Time (s)", row=num_plots, col=1)
    return fig, num_plots

# --- Gemini API 連携機能 ---
def create_summary_for_gemini(laps_data, lap_labels):
    if len(laps_data) < 2: return None
    file1_lap = laps_data[0]; file2_lap = laps_data[1]; label1 = lap_labels[0]; label2 = lap_labels[1]
    summary = [f"比較対象: {label1} vs {label2}"]
    metrics = {'Speed_kmh': '速度(km/h)', 'Throttle_pct': 'スロットル(%)', 'Lean_Angle': 'リーンアングル(度)', 'RPM': 'RPM', 'LAF1': 'ラムダ値', 'PBKF': 'フロントブレーキ圧', 'GYPIAN': 'ピッチ角', 'TRQPCTRD': 'トルクカット率', 'SLIP': 'スリップ'}
    for key, name in metrics.items():
        if key in file1_lap.columns and key in file2_lap.columns:
            summary.append(f"- {name} 平均: {label1}={file1_lap[key].mean():.2f}, {label2}={file2_lap[key].mean():.2f}")
    if 'GYXAC' in file1_lap.columns: summary.append(f"- 最大減速G: {label1}={-file1_lap['GYXAC'].min():.2f}G, {label2}={-file2_lap['GYXAC'].min():.2f}G")
    return "\n".join(summary)
def generate_gemini_report(summary_text):
    try: api_key = st.secrets["GEMINI_API_KEY"]; genai.configure(api_key=api_key)
    except (KeyError, AttributeError): st.error("Gemini APIキーが設定されていません。`.streamlit/secrets.toml`を確認してください。"); return None
    model = genai.GenerativeModel('gemini-1.5-flash'); prompt = f"あなたは優秀なバイクレースのデータエンジニア兼コーチです。\n以下の走行データサマリーを分析し、プロの視点から具体的なフィードバックを生成してください。\nデータは2つの走行データを比較したものです。\n\n# 走行データサマリー\n{summary_text}\n\n# 分析と提言\n上記のデータに基づき、以下のフォーマットで分析結果を記述してください。\nライダーと技術者の両方の視点から、良かった点と課題点を明確に分け、改善策も具体的に示してください。\n特に、燃料マップ(A/F, LAF1)、電子制御(TRQPCTRD, SLIP)、サスペンション(Pitch, G)に関する提言があれば、技術者の課題点に含めてください。\n\n---\n### 総合分析レポート\n\n#### ライダーの分析\n**[良かった点]**\n- \n\n**[課題点と改善案]**\n- \n\n#### 技術者（マシンセッティング）の分析\n**[良かった点]**\n- \n\n**[課題点と改善案]**\n- \n\n#### チーム全体の総括\n**[総括]**\n- \n---"
    try: response = model.generate_content(prompt); return response.text
    except Exception as e: st.error(f"Gemini APIの呼び出し中にエラーが発生しました: {e}"); return None

# -------------------------------------------------------------
#               メインの実行部分
# -------------------------------------------------------------
if 'df_input' in locals() and df_input is not None:
    df_processed = process_race_data(df_input)
    laps_to_plot, lap_labels = get_comparison_laps(df_processed, label1, label2)
    
    if laps_to_plot:
        with st.sidebar:
            st.header("⏱️ 比較セッション")
            st.success(f"**{label1}** vs **{label2}** の比較データを表示中。")
        
        col1, col2 = st.columns([6, 4], gap="large")
        with col1:
            st.subheader("📊 データ比較グラフ")
            with st.container(border=True):
                fig_plotly, num_plots = create_plotly_chart(laps_to_plot, lap_labels, df_processed.columns)
                if fig_plotly:
                    st.plotly_chart(fig_plotly, use_container_width=True)
                else:
                    st.warning("グラフを描画するためのデータ列がCSVファイルに不足しています。サイドバーの「デバッグ情報」で列名を確認してください。")
            st.markdown("---")
            st.subheader("🤖 AIによる分析レポート")
            with st.container(border=True):
                if st.button("分析レポートを生成する", key="gemini_button", type="primary", use_container_width=True):
                    summary = create_summary_for_gemini(laps_to_plot, lap_labels)
                    if summary:
                        with st.spinner("Geminiが分析中..."):
                            report = generate_gemini_report(summary)
                            if report: st.success("分析レポートが生成されました！"); st.markdown(report)
                    else:
                        st.warning("レポートを生成するための比較データが不足しています。")
        with col2:
            st.subheader("🗺️ 走行ラインマップ")
            
            with st.container(border=True):
                st.markdown("##### 🗺️ マーカー表示フィルター")
                marker_options = {"200m毎の距離": "distance", "TC介入 (34)": "tc", "Anti-jerk介入 (81)": "antijerk", "AFR": "ideal_af", "リーンアングル": "deep_lean"}
                selected_marker_labels = st.multiselect("表示するマーカーの種類を選択:", options=list(marker_options.keys()), default=["200m毎の距離", "リーンアングル"])
                selected_marker_keys = [marker_options[label] for label in selected_marker_labels]

            with st.container(border=True):
                if 'Latitude' in df_processed.columns and 'Longitude' in df_processed.columns:
                    map_center = [df_processed['Latitude'].mean(), df_processed['Longitude'].mean()]
                    m = folium.Map(location=map_center, zoom_start=17, tiles='OpenStreetMap')
                    lap_colors_map = ['#FF0000', '#0000FF']
                    
                    for i, lap_data in enumerate(laps_to_plot):
                        legend_label = lap_labels[i]; color = lap_colors_map[i % len(lap_colors_map)]
                        parent_fg = folium.FeatureGroup(name=legend_label, show=True).add_to(m)
                        points = list(zip(lap_data['Latitude'], lap_data['Longitude']))
                        folium.PolyLine(points, color=color, weight=4, opacity=0.7).add_to(parent_fg)
                        
                        if "distance" in selected_marker_keys and 'Lap_Distance' in lap_data.columns:
                            last_marker_distance = -200
                            for _, row in lap_data.iterrows():
                                if row['Lap_Distance'] >= last_marker_distance + 200:
                                    icon = folium.DivIcon(html=f'<div style="font-size: 9pt; color: gray; font-weight: 600;">{int(row["Lap_Distance"])}</div>')
                                    folium.Marker(location=[row['Latitude'], row['Longitude']], icon=icon, tooltip=f"Dist: {row['Lap_Distance']:.0f} m").add_to(parent_fg)
                                    last_marker_distance = row['Lap_Distance']
                        
                        if "tc" in selected_marker_keys and 'TC_Intervention' in lap_data.columns:
                            tc_zones = lap_data[lap_data['TC_Intervention'] == 34]
                            for _, row in tc_zones.iterrows():
                                folium.CircleMarker(location=(row['Latitude'], row['Longitude']), radius=5, color="orange", fill=True, tooltip=f"TC @ {row.get('Lap_Distance', row['Time']):.0f}m").add_to(parent_fg)

                        if "antijerk" in selected_marker_keys and 'TC_Intervention' in lap_data.columns:
                            aj_zones = lap_data[lap_data['TC_Intervention'] == 81]
                            for _, row in aj_zones.iterrows():
                                folium.CircleMarker(location=(row['Latitude'], row['Longitude']), radius=5, color="cyan", fill=True, tooltip=f"Anti-jerk @ {row.get('Lap_Distance', row['Time']):.0f}m").add_to(parent_fg)

                        if "ideal_af" in selected_marker_keys:
                            af_col_name = 'LAF1' if 'LAF1' in lap_data.columns else 'AFR'
                            if af_col_name in lap_data.columns:
                                afr_series = lap_data[af_col_name].copy()
                                is_laf = (af_col_name == 'LAF1')
                                if is_laf: afr_series *= 14.7
                                ideal_af_zones = lap_data[(afr_series >= 13.0) & (afr_series <= 13.9)]
                                for _, row in ideal_af_zones.iterrows():
                                    folium.CircleMarker(location=(row['Latitude'], row['Longitude']), radius=4, color="purple", fill=True, tooltip=f"AFR: {afr_series.loc[row.name]:.2f}").add_to(parent_fg)
                        
                        if "deep_lean" in selected_marker_keys and 'Lean_Angle' in lap_data.columns and not lap_data['Lean_Angle'].empty:
                            right_lean = lap_data[lap_data['Lean_Angle'] > 0]
                            if not right_lean.empty:
                                max_lean_right = right_lean['Lean_Angle'].max()
                                deep_lean_right = right_lean[right_lean['Lean_Angle'] >= max_lean_right - 20]
                                for _, row in deep_lean_right.iterrows():
                                    folium.CircleMarker(location=(row['Latitude'], row['Longitude']), radius=3, color=color, fill=False, tooltip=f"Lean: {row['Lean_Angle']:.1f}°").add_to(parent_fg)
                                peak_right_point = right_lean.loc[right_lean['Lean_Angle'].idxmax()]
                                folium.Marker(location=(peak_right_point['Latitude'], peak_right_point['Longitude']), icon=folium.Icon(color='red', icon='star'), tooltip=f"Max Right Lean: {max_lean_right:.1f}°").add_to(parent_fg)
                            left_lean = lap_data[lap_data['Lean_Angle'] < 0]
                            if not left_lean.empty:
                                max_lean_left = left_lean['Lean_Angle'].min()
                                deep_lean_left = left_lean[left_lean['Lean_Angle'] <= max_lean_left + 20]
                                for _, row in deep_lean_left.iterrows():
                                    folium.CircleMarker(location=(row['Latitude'], row['Longitude']), radius=3, color=color, fill=False, tooltip=f"Lean: {row['Lean_Angle']:.1f}°").add_to(parent_fg)
                                peak_left_point = left_lean.loc[left_lean['Lean_Angle'].idxmin()]
                                folium.Marker(location=(peak_left_point['Latitude'], peak_left_point['Longitude']), icon=folium.Icon(color='blue', icon='star'), tooltip=f"Max Left Lean: {max_lean_left:.1f}°").add_to(parent_fg)

                    folium.LayerControl(collapsed=False).add_to(m)
                    _, num_plots = create_plotly_chart(laps_to_plot, lap_labels, df_processed.columns)
                    map_height = 250 * num_plots if 'num_plots' in locals() and num_plots > 0 else 600
                    st_folium(m, width=None, height=map_height, key="folium_map_final")
                else:
                    st.info("GPSデータ (Latitude, Longitude) がCSVファイルにないため、地図は表示できません。")
    else:
        st.info("サイドバーから比較データを含むCSVファイルを1つアップロードしてください。")