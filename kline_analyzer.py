import cv2
import numpy as np


def analyze_kline_image(img_cv):
    img_draw = img_cv.copy()
    height, width = img_cv.shape[:2]

    # 图片尺寸校验
    if height <= 10 or width <= 10:
        return "⚠️图片尺寸过小，无法识别K线", img_draw

    h1 = int(height * 0.52)
    h2 = int(height * 0.68)
    h3 = int(height * 0.84)
    h1 = np.clip(h1, 1, height - 1)
    h2 = np.clip(h2, h1 + 1, height - 1)
    h3 = np.clip(h3, h2 + 1, height - 1)

    main_area = img_cv[0:h1, :]
    main_draw_area = img_draw[0:h1, :]
    vol_area = img_cv[h1:h2, :]
    macd_area = img_cv[h2:h3, :]
    rsi_area = img_cv[h3:height, :]

    # ==========1.K线红蓝比例 ==========
    try:
        hsv_main = cv2.cvtColor(main_area, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 120, 80])
        upper_red1 = np.array([10, 255, 255])
        mask_red1 = cv2.inRange(hsv_main, lower_red1, upper_red1)
        lower_red2 = np.array([170, 120, 80])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(mask_red1, cv2.inRange(hsv_main, lower_red2, upper_red2))
        red_k_pixel = cv2.countNonZero(mask_red)

        lower_blue = np.array([90, 100, 80])
        upper_blue = np.array([120, 255, 255])
        mask_blue = cv2.inRange(hsv_main, lower_blue, upper_blue)
        blue_k_pixel = cv2.countNonZero(mask_blue)

        total_k_pixel = red_k_pixel + blue_k_pixel
        ratio_red = red_k_pixel / total_k_pixel if total_k_pixel > 0 else 0
        ratio_blue = blue_k_pixel / total_k_pixel if total_k_pixel > 0 else 0
    except Exception as e:
        return f"【K线颜色识别失败】{str(e)}", img_draw

    # ==========2. MA均线 ==========
    try:
        hsv_main = cv2.cvtColor(main_area, cv2.COLOR_BGR2HSV)
        lower_green_ma = np.array([45, 120, 80])
        upper_green_ma = np.array([75, 255, 255])
        mask_ma5 = cv2.inRange(hsv_main, lower_green_ma, upper_green_ma)
        lower_yellow_ma = np.array([20, 120, 80])
        upper_yellow_ma = np.array([35, 255, 255])
        mask_ma20 = cv2.inRange(hsv_main, lower_yellow_ma, upper_yellow_ma)

        ma_w = mask_ma5.shape[1]
        w_half = int(ma_w * 0.5)
        w_half = np.clip(w_half, 0, ma_w - 1)

        ma5_right = mask_ma5[:, w_half:]
        ma20_right = mask_ma20[:, w_half:]

        m5 = cv2.moments(ma5_right)
        m20 = cv2.moments(ma20_right)
        ma5_y = m5["m01"] / m5["m00"] if m5["m00"] > 0 else None
        ma20_y = m20["m01"] / m20["m00"] if m20["m00"] > 0 else None

        ma_score = 20
        ma_desc = "⚠️ 均线识别不完整；"
        ma_bull = False

        if ma5_y is not None and ma20_y is not None:
            ma_w2 = mask_ma5.shape[1]
            w_right = int(ma_w2 * 0.75)
            w_right = np.clip(w_right, w_half, ma_w2 - 1)

            ma20_tail = mask_ma20[:, w_right:]
            m20_tail = cv2.moments(ma20_tail)
            ma20_tail_y = m20_tail["m01"] / m20_tail["m00"] if m20_tail["m00"] > 0 else None
            ma20_head = mask_ma20[:, w_half:w_right]
            m20_head = cv2.moments(ma20_head)
            ma20_head_y = m20_head["m01"] / m20_head["m00"] if m20_head["m00"] > 0 else None

            if ma5_y < ma20_y:
                ma_score = 10
                ma_desc = "❌ MA5在MA20下方，短线死叉；"
                ma_bull = False
            else:
                ma_bull = True
                if ma20_tail_y is not None and ma20_head_y is not None and ma20_tail_y > ma20_head_y:
                    ma_score = 28
                    ma_desc = "🟡 MA5上穿MA20短线金叉，但MA20中期向下，属于下跌趋势反弹金叉；"
                else:
                    ma_score = 40
                    ma_desc = "✅ MA5上穿MA20，MA20向上，趋势反转金叉；"
    except Exception as e:
        return f"【均线识别失败】{str(e)}", img_draw

    # ==========3. VOL成交量 ==========
    try:
        vol_w = vol_area.shape[1]
        vol_w_split = int(vol_w * 0.75)
        vol_w_split = np.clip(vol_w_split, 0, vol_w - 1)
        vol_history_region = vol_area[:, :vol_w_split]
        vol_right_region = vol_area[:, vol_w_split:]

        hsv_vol_his = cv2.cvtColor(vol_history_region, cv2.COLOR_BGR2HSV)
        vol_his_mask = cv2.inRange(hsv_vol_his, np.array([0, 30, 60]), np.array([180, 255, 255]))
        vol_his_px = cv2.countNonZero(vol_his_mask)
        avg_his_px = vol_his_px / 45

        hsv_vol_now = cv2.cvtColor(vol_right_region, cv2.COLOR_BGR2HSV)
        vol_now_mask = cv2.inRange(hsv_vol_now, np.array([0, 30, 60]), np.array([180, 255, 255]))
        vol_now_px = cv2.countNonZero(vol_now_mask)

        vol_score = 0
        vol_desc = ""
        vol_strong = False
        if vol_now_px > avg_his_px * 1.35:
            vol_score = 25
            vol_desc = "✅ 最新K线成交量放量，资金确认；"
            vol_strong = True
        elif vol_now_px > avg_his_px * 0.7:
            vol_score = 15
            vol_desc = "🟡 成交量和近期持平；"
            vol_strong = False
        else:
            vol_score = 8
            vol_desc = "❌ 最新K线成交量萎缩，反弹缺少资金；"
            vol_strong = False
    except Exception as e:
        return f"【成交量识别失败】{str(e)}", img_draw

    # ==========4. MACD ==========
    try:
        macd_w = macd_area.shape[1]
        macd_w_split = int(macd_w * 0.75)
        macd_w_split = np.clip(macd_w_split, 0, macd_w - 1)
        macd_right_region = macd_area[:, macd_w_split:]
        hsv_macd = cv2.cvtColor(macd_right_region, cv2.COLOR_BGR2HSV)

        mask_macd_red1 = cv2.inRange(hsv_macd, np.array([0, 100, 60]), np.array([10, 255, 255]))
        mask_macd_red2 = cv2.inRange(hsv_macd, np.array([170, 100, 60]), np.array([180, 255, 255]))
        mask_macd_red = cv2.bitwise_or(mask_macd_red1, mask_macd_red2)
        macd_red_px = cv2.countNonZero(mask_macd_red)
        mask_macd_green = cv2.inRange(hsv_macd, np.array([45, 100, 60]), np.array([75, 255, 255]))
        macd_green_px = cv2.countNonZero(mask_macd_green)

        macd_score = 0
        macd_desc = ""
        macd_bull = False
        h_macd = macd_right_region.shape[0]
        mid_h = int(h_macd * 0.5)
        mask_line = cv2.inRange(hsv_macd, np.array([18, 90, 90]), np.array([38, 255, 255]))
        line_y_sum = 0
        line_cnt = 0
        for y in range(h_macd):
            cnt = np.count_nonzero(mask_line[y, :])
            if cnt > 0:
                line_y_sum += y * cnt
                line_cnt += cnt
        dif_dea_avg_y = line_y_sum / line_cnt if line_cnt > 0 else None
        under_zero = False
        if dif_dea_avg_y is not None:
            under_zero = dif_dea_avg_y > mid_h

        if macd_red_px > macd_green_px:
            if under_zero:
                macd_score = 15
                macd_desc = "🟡 MACD红柱占优，但DIF/DEA整体在0轴下方，空头环境弱反弹；"
                macd_bull = False
            else:
                macd_score = 35
                macd_desc = "✅ MACD红柱占优，DIF/DEA位于0轴上方，多头动量；"
                macd_bull = True
        else:
            macd_score = 10
            macd_desc = "❌ MACD绿柱占优，空头动量；"
            macd_bull = False
    except Exception as e:
        return f"【MACD识别失败】{str(e)}", img_draw

    # ==========5. 压力位、K点、止损线 ==========
    k_y_array = np.array([])
    resist_score = 0
    stop_loss_price_desc = ">⚠️未能识别K线轮廓，无法计算止损像素位置"
    try:
        mr_w = main_area.shape[1]
        cut_pos = int(mr_w * 0.7)
        cut_pos = np.clip(cut_pos, 0, mr_w - 1)
        main_right = main_area[:, cut_pos:]
        hsv_main_r = cv2.cvtColor(main_right, cv2.COLOR_BGR2HSV)
        mask_kline = cv2.inRange(hsv_main_r, np.array([0, 0, 50]), np.array([180, 255, 255]))
        k_points = np.where(mask_kline > 0)
        k_y_array = k_points[0]

        if len(k_y_array) > 0:
            recent_low_y = np.max(k_y_array)
            recent_high_y = np.min(k_y_array)
            cur_y = np.mean(k_y_array)
            long_stop_y = recent_low_y + (recent_high_y - recent_low_y) * 0.03
            short_stop_y = recent_high_y - (recent_high_y - recent_low_y) * 0.03
            stop_loss_price_desc = f">【图像像素参考，仅截图定位用，非真实行情价格】\n>做多止损参考位置：截图纵向像素Y={long_stop_y:.0f}（K线近期低点下方）\n>做空止损参考位置：截图纵向像素Y={short_stop_y:.0f}（K线近期高点上方）"

            if 0 <= long_stop_y < main_draw_area.shape[0]:
                cv2.line(main_draw_area, (0, int(long_stop_y)), (width, int(long_stop_y)), (0, 0, 255), 2)
                cv2.putText(main_draw_area, "做多止损", (20, int(long_stop_y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255), 2)
            if 0 <= short_stop_y < main_draw_area.shape[0]:
                cv2.line(main_draw_area, (0, int(short_stop_y)), (width, int(short_stop_y)), (255, 0, 0), 2)
                cv2.putText(main_draw_area, "做空止损", (20, int(short_stop_y) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 0), 2)

            max_y = np.min(k_y_array)
            min_y = np.max(k_y_array)
            cur_price_y = np.mean(k_y_array)
            price_range = min_y - max_y
            if price_range > 0 and (cur_price_y - max_y) / price_range < 0.35:
                resist_score = -12
    except Exception as e:
        return f"【K线轮廓识别失败】{str(e)}", img_draw

    # ==========6.影线识别 ==========
    shadow_score = 0
    shadow_desc = ""
    try:
        kl_w = main_area.shape[1]
        cut_pos = int(kl_w * 0.8)
        cut_pos = np.clip(cut_pos, 0, kl_w - 1)
        k_last_region = main_area[:, cut_pos:]
        hsv_k_last = cv2.cvtColor(k_last_region, cv2.COLOR_BGR2HSV)
        mask_k = cv2.inRange(hsv_k_last, np.array([0, 0, 40]), np.array([180, 255, 255]))
        k_y_pos = np.where(mask_k > 0)[0]
        if len(k_y_pos) > 0:
            k_top = np.min(k_y_pos)
            k_bottom = np.max(k_y_pos)
            body_h = abs(np.percentile(k_y_pos, 70) - np.percentile(k_y_pos, 30))
            full_h = k_bottom - k_top
            if body_h > 0 and full_h / body_h > 2.2:
                shadow_score = -10
                shadow_desc = "⚠️ 识别长上影K线，上方抛压较重，扣10分；"
            else:
                shadow_desc = "✅ K线无上影压力；"
        else:
            shadow_desc = "⚠️ K线影线识别失败"
    except Exception as e:
        return f"【影线识别失败】{str(e)}", img_draw

    # ==========7.连续K线 ==========
    continue_score = 0
    continue_desc = ""
    try:
        seq_w = main_area.shape[1]
        cut_pos = int(seq_w * 0.75)
        cut_pos = np.clip(cut_pos, 0, seq_w - 1)
        seq_region = main_area[:, cut_pos:]
        hsv_seq = cv2.cvtColor(seq_region, cv2.COLOR_BGR2HSV)
        mask_red_seq = cv2.inRange(hsv_seq, np.array([0, 100, 60]), np.array([10, 255, 255]))
        mask_blue_seq = cv2.inRange(hsv_seq, np.array([90, 100, 60]), np.array([120, 255, 255]))
        red_col_sum = np.count_nonzero(mask_red_seq, axis=0)
        blue_col_sum = np.count_nonzero(mask_blue_seq, axis=0)
        cnt_cont_red = 0
        cnt_cont_blue = 0
        for col in reversed(range(len(red_col_sum))):
            if red_col_sum[col] > 10:
                cnt_cont_red += 1
            else:
                break
        for col in reversed(range(len(blue_col_sum))):
            if blue_col_sum[col] > 10:
                cnt_cont_blue += 1
            else:
                break
        if cnt_cont_red >= 3:
            continue_score += 6
            continue_desc = f"✅ 连续{cnt_cont_red}根阳线，短线多头延续；"
        elif cnt_cont_blue >= 3:
            continue_score -= 6
            continue_desc = f"❌ 连续{cnt_cont_blue}根阴线，短线空头延续；"
        else:
            continue_desc = f"🟡 无连续3根以上K线，连续阳线:{cnt_cont_red}根，连续阴线:{cnt_cont_blue}根；"
    except Exception as e:
        return f"【连续K线识别失败】{str(e)}", img_draw

    # ==========8.RSI ==========
    rsi_score = 0
    rsi_desc = ""
    try:
        h_rsi = rsi_area.shape[0]
        rsi_mid = int(h_rsi * 0.5)
        hsv_rsi = cv2.cvtColor(rsi_area, cv2.COLOR_BGR2HSV)
        mask_rsi_line = cv2.inRange(hsv_rsi, np.array([15, 70, 70]), np.array([40, 255, 255]))
        rsi_line_y_sum = 0
        rsi_line_cnt = 0
        for y in range(h_rsi):
            c = np.count_nonzero(mask_rsi_line[y, :])
            if c > 0:
                rsi_line_y_sum += y * c
                rsi_line_cnt += c
        rsi_line_y = rsi_line_y_sum / rsi_line_cnt if rsi_line_cnt > 0 else None
        if rsi_line_y is not None:
            ratio_pos = rsi_line_y / h_rsi
            rsi_val_est = 100 - (ratio_pos * 100)
            if rsi_val_est >= 80:
                rsi_score -= 8
                rsi_desc = f"⚠️ RSI估算{rsi_val_est:.1f}，进入超买区间，多头风险加大，扣8分；"
            elif rsi_val_est <= 20:
                rsi_score += 8
                rsi_desc = f"✅ RSI估算{rsi_val_est:.1f}，进入超卖区间，存在反弹机会，加8分；"
            else:
                rsi_desc = f"🟡 RSI估算{rsi_val_est:.1f}，中性区间；"
        else:
            rsi_desc = "⚠️ RSI线条识别失败；"
    except Exception as e:
        return f"【RSI识别失败】{str(e)}", img_draw

    # =========总分 ==========
    total_score = ma_score + vol_score + macd_score + resist_score + shadow_score + continue_score + rsi_score
    total_score = np.clip(total_score, 0, 100)
    prob_up = np.clip(total_score, 10, 85)
    prob_down = 100 - prob_up
    prob_side = 0
    if 40 <= total_score <= 60:
        prob_side = 30
        prob_up = prob_up * 0.7
        prob_down = prob_down * 0.7

    # ==========操作建议 ==========
    suggest = ""
    reason_list = []
    long_condition = ma_bull and macd_bull and vol_strong and (not under_zero)
    short_condition = (not ma_bull) and (not macd_bull)

    if long_condition:
        suggest = "👉【操作建议：买涨（做多）】"
        reason_list.append(f"1.均线：MA5站上MA20，趋势多头信号({ma_score}/40分)")
        reason_list.append(f"2.MACD：MACD红柱大于绿柱，DIF/DEA位于0轴上方，上涨动量确认({macd_score}/35分)")
        reason_list.append(f"3.成交量：最新K线成交量放量，资金进场({vol_score}/25分)")
        if resist_score < 0:
            reason_list.append(f"⚠️价格靠近压力位，扣{abs(resist_score)}分，注意承压回落风险")
        if shadow_score < 0:
            reason_list.append(f"⚠️长上影抛压，扣{abs(shadow_score)}分")
        reason_list.append(f"{continue_desc}")
        reason_list.append(f"{rsi_desc}")
        reason_list.append(f"综合上涨概率{prob_up:.1f}% > 下跌概率{prob_down:.1f}%，共振多头信号。")
        reason_list.append("风控：严格设置止损，一旦MA5重新下穿MA20，考虑离场。")
        reason_list.append(stop_loss_price_desc)
    elif short_condition:
        suggest = "👉【操作建议：买跌（做空）】"
        reason_list.append(f"1.均线：MA5在MA20下方，短线死叉，均线空头信号({ma_score}/40分)")
        reason_list.append(f"2.MACD：MACD绿柱大于红柱，下跌动量占优({macd_score}/35分)")
        reason_list.append(f"3.成交量：最新K线成交量放量，资金出逃({vol_score}/25分)")
        reason_list.append(f"{continue_desc}")
        reason_list.append(f"{rsi_desc}")
        reason_list.append(f"综合下跌概率{prob_down:.1f}% > 上涨概率{prob_up:.1f}%，共振空头信号。")
        reason_list.append("风控：严格设置止损，一旦MA5上穿MA20，考虑离场。")
        reason_list.append(stop_loss_price_desc)
    else:
        suggest = "👉【操作建议：观望，不进场】"
        reason_list.append("数据原因：指标没有形成共振信号，属于单一指标信号，虚假信号概率高。")
        reason_list.append(
            f"当前得分{total_score:.0f}/100，上涨概率{prob_up:.1f}%，下跌概率{prob_down:.1f}%，震荡概率{prob_side:.1f}%")
        reason_list.append(
            "典型场景：均线金叉但是成交量萎缩/MACD处于0轴下方，属于下跌趋势中的反弹诱多，不建议追涨；反之同理。")
        reason_list.append("等待三项指标共振后再开仓，提高胜率。")
        reason_list.append(stop_loss_price_desc)
    reason_text = "\n".join(reason_list)

    report = f"""
==================== K线图像综合分析报告 ====================
【K线红蓝比例】上涨K线占比：{ratio_red:.2%}，下跌K线占比：{ratio_blue:.2%}
【MA均线分析】{ma_desc}
【成交量VOL分析】{vol_desc}
【MACD指标分析】{macd_desc}
【压力位评估】{"🟡价格靠近压力区域，扣减" + str(abs(resist_score)) + "分" if resist_score < 0 else "✅价格远离压力位"}
【K线影线分析】{shadow_desc}
【连续K线统计】{continue_desc}
【RSI指标分析】{rsi_desc}

👉 综合打分：{total_score:.0f}/100
👉 当下概率预测：
短线上涨概率：{prob_up:.1f}%
短线下跌概率：{prob_down:.1f}%
震荡横盘概率：{prob_side:.1f}%

{suggest}
【数据支撑理由】
{reason_text}

====================风控提醒====================
1. 1分钟周期信号变化极快，信号随时失效，每一笔单子必须带止损。
2. 本程序是截图图像识别，存在颜色识别误差，不能替代行情原始数据。
3. 期货杠杆高，行情会受外盘、消息、资金冲击，技术信号经常失效。

⚠️免责声明：图像识别技术分析仅为学习研究用途，**不构成任何交易建议**。期货交易风险巨大，盈亏全部由交易者自行承担。
"""
    return report, img_draw
