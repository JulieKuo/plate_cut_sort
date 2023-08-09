import base64, json, random
from traceback import format_exc
from rectpack import newPacker
import pandas as pd
import numpy as np


def get_input(argv, logging):
    try:   
        if len(argv) == 2:
            input_ = argv[1]
            input_ = base64.b64decode(input_).decode('utf-8')
            input_ = json.loads(input_)

            return input_
        else:
            logging.info("Input parameter error.")
    except:
        logging.error(format_exc())


def cutting(width_limit, height_limit, rectangles):

    bins = [[width_limit, height_limit]] # 設定箱子大小

    packer = newPacker()

    # Add the rectangles to packing queue
    for r in rectangles:
        packer.add_rect(*r)

    # Add the bins where the rectangles will be placed
    for b in bins:
        packer.add_bin(*b)

    # Start packing
    packer.pack()

    return len(packer.rect_list()) # 箱子中的矩形數量


def calculate_waste(df, width_limit, height_limit):
    # 每次加入一塊矩形，並計算箱子的耗損率

    df[["waste", "plate_id"]] = [1, 0]
    cut_items = df[["length", "width"]]
    cut_items = cut_items.to_numpy() # 轉換成newPacker可接收的格式

    start = 0 # 第n個箱子的第一個矩形於items中的index
    end   = 0 # 第n個箱子的最後一個矩形於items中的index
    plate_id = 1 # 箱子id
    accum_areas = 0 # 第n個箱子的累績使用面積
    while end < len(cut_items):
        rectangles = cut_items[start:end+1] # 欲放入箱子中的所有矩形
        length = cutting(width_limit, height_limit, rectangles) # 查看是否可全數放入箱子中，如果可放入，箱子中的矩形數量會加1
        accum_num = end - start + 1 # 目前測試放入箱中的矩形數量
        accum_areas += cut_items[end][0] * cut_items[end][1] # 加入該矩形後，箱子的累績使用面積 (長 * 寬)
        waste = 1 - (accum_areas / (width_limit * height_limit)) # 箱子的耗損率
        # print(f"{start} ~ {end}, length = {length}, accum_num = {accum_num}, waste = {waste}, plate_id = {plate_id}")

        df.loc[end, "waste"] = waste
        df.loc[end, "plate_id"] = plate_id
        if (length != accum_num): # 如果欲放入箱中的矩形數量和經過cutting後箱中的矩形數量不同，表示該矩形無法放入箱中，新增一個箱子，並重設狀態
            start = end
            plate_id += 1
            accum_areas = 0
        else:
            end += 1
            if (end < len(cut_items) - 1) and (df.loc[end, "color"] != df.loc[end - 1, "color"]): # 如果下一個矩形的顏色不同，則新增一個箱子，並重設狀態
                start = end
                plate_id += 1
                accum_areas = 0

    return df


def calculate_mean_wast(df, waste_json = None):
    # 計算板材使用數和平均板材耗損率
    plate_g = df.groupby("plate_id")
    plate_g_size = plate_g.size()
    plate_df = pd.DataFrame()
    for group in plate_g_size.index:
        df_g = plate_g.get_group(group)
        plate_df = pd.concat([plate_df, df_g.iloc[[-1]]]) # 每次把某個plate_id的最後一個耗損率存進plate_df

    plate_num = len(plate_df) # 板材使用數
    mean_waste = plate_df["waste"].mean().round(4) # 平均板材耗損率


    # 逐顏色計算板材使用數和平均板材耗損率
    color_g = plate_df.groupby("color")
    color_g_size = color_g.size()
    color_df = pd.DataFrame()
    color = {}
    for group in color_g_size.index:
        df_g = color_g.get_group(group)
        color_df = pd.concat([color_df, df_g.iloc[:-1]]) # 刪除每種顏色最後一片並存入存進color_df

        color[group] = {
            "waste": df_g["waste"].mean().round(4),
            "plate": len(df_g)
            }


    mean_waste_rm_last = color_df["waste"].mean().round(4) # 刪除每種顏色最後一片後的平均板材耗損率
    plate_num_rm_last = len(color_df) # 刪除每種顏色最後一片後的板材使用數


    waste_score = {
        "plate":{        
            "waste": mean_waste_rm_last,
            "plate": plate_num_rm_last,
            "no_rm_waste": mean_waste,    
            "no_rm_plate": plate_num
        },
        "color": color
    }


    if waste_json:
        with open(waste_json, 'w', encoding='utf-8') as f:
            json.dump(waste_score, f, indent = 4, ensure_ascii=False)


    return waste_score


def calculate_num(col, df, target, flag = "train"):
    df[col] = df[col].fillna("None")
    if len(col) == 1:
        col = col[0]
    group_counts = df[col].value_counts().to_dict() # 計算col底下有多少個資料(即商品)，並轉為dict
    index_comb = df.set_index(col).index # 設定item的index為col
    group_counts_map = index_comb.map(group_counts) # 用map對照返回相對應的值
    new_col = f"{target}_num"
    df[new_col] = group_counts_map # 新增group_counts至item中

    # 計算每個col中第n個cabinet(item)佔總cabinet(item)數量的比值
    accum_counts = {key: 0 for key in group_counts.keys()} # 建立一個dict，內部儲存初始化的累計次數

    if flag == "train":
        for i, index_comb in enumerate(index_comb):
            accum_counts[index_comb] += 1
            df.loc[i, new_col] = accum_counts[index_comb] / df.loc[i, new_col]
        
        return df
    else:
        return df, accum_counts
    

def generate_label(df, target):
    # 下一個plate是否相同
    df["same_plate"] = False
    for i in range(0, len(df)-1):
        if df.loc[i, "plate_id"] != df.loc[i+1, "plate_id"]:
            df.loc[i, "same_plate"] = False
        else:
            df.loc[i, "same_plate"] = True

    df["same_cabinet"] = df["cabinet"] == df["cabinet"].shift(-1) # 下一個cabinet是否相同
    df["same_item"]    = (df["length"] == df["length"].shift(-1)) & (df["width"] == df["width"].shift(-1)) # 下一個length、width是否相同

    target0_index = df[df["same_plate"] & df["same_cabinet"] & df["same_item"]].index # 同plate，同cabinet，同item
    target1_index = df[df["same_plate"] & df["same_cabinet"] & (~ df["same_item"])].index # 同plate，同cabinet，不同item
    target2_index = df[df["same_plate"] & (~ df["same_cabinet"]) & df["same_item"]].index # 同plate，不同cabinet，同item
    target3_index = df[df["same_plate"] & (~ df["same_cabinet"]) & (~ df["same_item"])].index # 同plate，不同cabinet，不同item
    target4_index = df[(~ df["same_plate"])].index # 因color或空間不足而換新plate，找長寬最大的item

    df.insert(0, target, None)
    df.loc[target0_index, target] = 0
    df.loc[target1_index, target] = 1
    df.loc[target2_index, target] = 2
    df.loc[target3_index, target] = 3
    df.loc[target4_index, target] = 4

    df[target] = df[target].astype(int)
    df = df.drop(["same_plate", "same_cabinet", "same_item", "plate_id"], axis = 1)

    return df


def target4(test, weights):
    # 換新版材，找area最大的item
    next_df = test[test["selected"] == 0] # 抓出可選擇的板材
    if len(next_df) > 0:
        date   = next_df.sort_values(["order_id", "e_ship_date", "area", "length", "width"], ascending = [True, True, False, False, False]).index[0] # 選取日期最小的產品做為下一個切割物
        length = next_df.sort_values(["order_id", "length", "area", "width", "e_ship_date"], ascending = [True, False, False, False, True]).index[0] # 選取長度最大的產品做為下一個切割物
        width  = next_df.sort_values(["order_id", "width", "area", "length", "e_ship_date"], ascending = [True, False, False, False, True]).index[0] # 選取寬度最大的產品做為下一個切割物
        area   = next_df.sort_values(["order_id", "area", "length", "width", "e_ship_date"], ascending = [True, False, False, False, True]).index[0] # 選取面積最大的產品做為下一個切割物
        next_index = random.choices(population = [date, length, width, area], weights = weights, k = 1)[0]
        
        return next_index
    else:
        return None


def target0_3(test, col_dict, num, index, weights, flag = False):
    col  = col_dict[num] # 抓出欲匹配的項目
    item = test.loc[index, col] # 抓出欲匹配的項目的值
    same_data = (item.to_numpy() == test[col].to_numpy()).all(axis = 1) # 確認板材是否匹配
    selected  = test["selected"] == 0 # 確認板材是否已被選擇
    next_df = test[(same_data & selected)] # 抓出可選擇的板材

    # 依序尋找可用的候選板材，按間隔抽出10筆資料嘗試
    if flag:
        next_df = next_df.sort_values(["order_id", "area", "length", "width"], ascending = [True, False, False, False])
        next_df = next_df.drop_duplicates(subset = ["length", "width"]) # 只保留所有候選組合的第一個，節省搜索時間
        if len(next_df) > 0:
            test_index = list(set(np.linspace(start = 0, stop = len(next_df)-1, num = 10, dtype = int))) # 只抓出10個
            next_df = next_df.iloc[test_index]
        next_df = next_df[next_df["fail"] == 0] # 刪除已被嘗試過的資料

    if len(next_df) > 0:
        date    = next_df.sort_values(["order_id", "e_ship_date", "area", "length", "width"], ascending = [True, True, False, False, False]).index[0] # 選取日期最小的產品做為下一個切割物
        length  = next_df.sort_values(["order_id", "length", "area", "width", "e_ship_date"], ascending = [True, False, False, False, True]).index[0] # 選取長度最大的產品做為下一個切割物
        width   = next_df.sort_values(["order_id", "width", "area", "length", "e_ship_date"], ascending = [True, False, False, False, True]).index[0] # 選取寬度最大的產品做為下一個切割物
        area    = next_df.sort_values(["order_id", "area", "length", "width", "e_ship_date"], ascending = [True, False, False, False, True]).index[0] # 選取面積最大的產品做為下一個切割物
        next_index = random.choices(population = [date, length, width, area], weights = weights, k = 1)[0]
        
        return next_index
    else:
        return None


def get_next_plate(col_dict, index, weights, df, length_limit, width_limit):
    num = df.loc[index, "label"]

    # target為0~3時，尋找到可放入的的板材
    if num < 4:
        df["fail"] = 0
        next_index = True
        # 依序尋找候選板材，直到找不到下一個next_index，或找到可放入箱中的板材
        while next_index:
            next_index = target0_3(df, col_dict, num, index, weights, flag = True) # 尋找下一個板材的index

            if next_index:
                df.loc[next_index, "fail"] = 1 # 紀錄候選板材是否已嘗試放入過
                df.iloc[index+1], df.iloc[next_index] = df.iloc[next_index], df.iloc[index+1] # 第index個板材和第next_index個板材交換
            
                # 判斷箱子的空間是否足夠
                df.loc[index+1, "plate_id"] = df.loc[index, "plate_id"]
                cut_items = df[df["plate_id"] == df.loc[index, "plate_id"]] # 抓出與目前這塊板材相同plate_id的其他板材
                rectangles = cut_items[["length", "width"]].to_numpy() # 轉換為可放入cutting函式中的newPacker套件之資料格式
                length = cutting(length_limit, width_limit, rectangles) # 計算箱子內有多少矩形

                # 如果箱子(原始板材)中的矩形(切割的板材)數量與欲放入的矩形數量相同，表示空間足夠，跳出迴圈
                if len(rectangles) == length:
                    df.loc[index+1, "accum_areas"] = df.loc[index, "accum_areas"] + df.loc[index+1, "area"] # 箱中的累積使用面積
                    df.loc[index+1, "waste"] = (1 - (df.loc[index+1, "accum_areas"] / (length_limit * width_limit))) # 計算waste欄位，及耗損率
                    break
                else:
                    df.loc[index+1, "plate_id"] = 0

    # target為0~2時，若未找到可放入的的板材，則以限制最寬鬆的target=3求next_index
    if (df.loc[index+1, "plate_id"] == 0) and (num < 3):
        df["fail"] = 0
        next_index = target0_3(df, col_dict, 3, index, weights) # 尋找下一個板材的index
        if next_index:
            df.iloc[index+1], df.iloc[next_index] = df.iloc[next_index], df.iloc[index+1] # 第index個板材和第next_index個板材交換
            df.loc[index+1, "plate_id"] = df.loc[index, "plate_id"] + 1
            df.loc[index+1, "accum_areas"] = df.loc[index+1, "area"] # 箱中的累積使用面積
            df.loc[index+1, "waste"] = (1 - (df.loc[index+1, "accum_areas"] / (length_limit * width_limit))) # 計算waste欄位，及耗損率

    # target=4或未找到可放入的的板材時，換新的箱子，以target=4求next_index
    if (df.loc[index+1, "plate_id"] == 0):
        next_index = target4(df, weights) # 尋找下一個板材的index
        df.iloc[index+1], df.iloc[next_index] = df.iloc[next_index], df.iloc[index+1] # 第index個板材和第next_index個板材交換
        df.loc[index+1, "plate_id"] = df.loc[index, "plate_id"] + 1
        df.loc[index+1, "accum_areas"] = df.loc[index+1, "area"] # 箱中的累積使用面積
        df.loc[index+1, "waste"] = (1 - (df.loc[index+1, "accum_areas"] / (length_limit * width_limit))) # 計算waste欄位，及耗損率
    

    return df


def error(logging, message, model_id):
    logging.error(message)
    result = {
        "status": "fail",
        "reason": message,
        "model_id": model_id
        }
    
    return result