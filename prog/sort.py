import os, json
import pandas as pd
from traceback import format_exc
from log_config import Log
from db import Database
from tools import *
from tqdm import tqdm


class Sort():
    def __init__(self, root, logging):
        self.logging = logging

        # 取得output位置
        clean_path = os.path.join(root, "data", "sort")        
        os.makedirs(clean_path, exist_ok = True)
        self.output_csv = os.path.join(clean_path, "data.csv")
        self.output_json = os.path.join(clean_path, "output.json")        
        self.waste_json = os.path.join(clean_path, "waste.json")

        # 取得config        
        config_path = os.path.join(root, "prog", "config.json")
        self.logging.info(f'Read config from {config_path}')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
    
    def data_sort(self):
        try:
            self.logging.info(f'Get data from database.')
            db_con = self.config["database"]
            db = Database(db_con)

            df = db.get_data(db_con["table"]["predict"])
            remove = (df[["length", "width"]] <= 0).any(axis = 1) # 長寬為0無法切割
            df = df[~remove].reset_index(drop = True)


            self.logging.info('Initialize sort.')
            df = df.sort_values(["color", "order_id", "cabinet", "length", "width"], ascending = [True, True, True, False, False])
            df = df.reset_index(drop = True)


            self.logging.info('Calculate area.')
            df["area"] = df["length"] * df["width"] # 每個矩形的面積
            length_limit = self.config["limit"]["length"]
            width_limit = self.config["limit"]["width"]
            area_limit = length_limit * width_limit
            df["area_prob"] = df["area"] / area_limit # 矩形站箱子的面積


            self.logging.info('Calculate waste and plate_id & Sort.')
            df[["waste", "plate_id"]] = [1, 0]

            start = 0 # 第n個箱子的第一個矩形於items中的index
            end = 0 # 第n個箱子的最後一個矩形於items中的index
            plate_id = 1 # 箱子id
            accum_areas = 0 # 第n個箱子的累績使用面積

            # 每次加入一塊矩形，並計算箱子的耗損率
            pbar = tqdm(total = len(df), ncols = 150)
            while end < len(df):
                cut_items = df[["length", "width"]].to_numpy() # 轉換成newPacker可接收的格式
                rectangles = cut_items[start:end+1] # 欲放入箱子中的所有矩形
                length = cutting(length_limit, width_limit, rectangles) # 查看是否可全數放入箱子中，如果可放入，箱子中的矩形數量會加1
                accum_num = end - start + 1 # 目前測試放入箱中的矩形數量

                # 如果欲放入箱中的矩形數量和經過cutting後箱中的矩形數量不同，表示該矩形無法放入箱中，尋找其他可放入的矩形
                if (length != accum_num):
                    init = df.iloc[end].copy()
                    color, _, _ = df.loc[end, ["color", "length", "width"]]
                    df1 = df.query("(color == @color) and (plate_id == 0)") # 抓出候選組合
                    df1_index = df1[["length", "width"]].drop_duplicates(keep = "first").index # 只保留所有候選組合的第一個，節省搜索時間
                    
                    # 依序尋找所有可能的解
                    for i, replace in enumerate(df1_index):
                        pbar.set_postfix({"plate_id": plate_id, "waste": waste, "test_num": f"{i+1}/{len(df1_index)}"}) # 更新目前搜索進度
                        pbar.update(0)

                        df.iloc[end] = df.iloc[replace]
                        cut_items = df[["length", "width"]].to_numpy() # 轉換成newPacker可接收的格式
                        rectangles = cut_items[start:end+1] # 欲放入箱子中的所有矩形
                        length = cutting(length_limit, width_limit, rectangles) # 查看是否可全數放入箱子中，如果可放入，箱子中的矩形數量會加1
                        
                        # 有解則跳出迴圈
                        if (length == accum_num):
                            df.iloc[replace] = init
                            break
                    
                    # 無可放入的矩形，新增一個箱子，並重設狀態
                    if (length != accum_num):
                        df.iloc[end] = init
                        start = end
                        plate_id += 1
                        accum_areas = 0
                        continue

                # 放入矩形進箱子裡
                accum_areas += cut_items[end][0] * cut_items[end][1] # 加入該矩形後，箱子的累績使用面積 (長 * 寬)
                waste = 1 - (accum_areas / (length_limit * width_limit)) # 箱子的耗損率
                # print(f"{start} ~ {end}, length = {length}, accum_num = {accum_num}, waste = {waste}, plate_id = {plate_id}")

                df.loc[end, "waste"] = waste
                df.loc[end, "plate_id"] = plate_id

                end += 1
                if (end < len(cut_items) - 1) and (df.loc[end, "color"] != df.loc[end - 1, "color"]): # 如果下一個矩形的顏色不同，則新增一個箱子，並重設狀態
                    start = end
                    plate_id += 1
                    accum_areas = 0

                pbar.set_postfix({"plate_id": plate_id, "waste": waste}) # 更新目前排序進度
                pbar.update(1)

            pbar.close()

            self.logging.info(f"Save waste to {self.waste_json}")
            calculate_mean_wast(df, self.waste_json)

            self.logging.info(f'Save data to {self.output_csv}')
            df = df.drop(["area", "area_prob", "waste", "plate_id"], axis = 1)
            df.to_csv(self.output_csv, encoding='utf-8-sig', index = False)

            result = {
                "status": "success",
                "data_counts": len(df),
                "remove_counts": sum(remove)
                }
                
        except:
            self.logging.error(format_exc())
            result = {
                "status": "fail",
                "reason": format_exc(),
                }

        finally:
            self.logging.info(f'Save output to {self.output_json}.')
            with open(self.output_json, 'w') as file:
                json.dump(result, file, indent = 4)



if __name__ == '__main__':
    # 取得根目錄
    current_path = os.path.abspath(__file__)
    prog_path = os.path.dirname(current_path)
    root = os.path.dirname(prog_path)


    log = Log()
    log_path = os.path.join(root, "logs")
    os.makedirs(log_path, exist_ok = True)
    logging = log.set_log(filepath = os.path.join(log_path, "sort.log"), level = 2, freq = "D", interval = 50, backup = 3, name = "sort")
    
    logging.info("-"*100)
    # logging.info(f"root: {root}")
    

    sort = Sort(root, logging)
    sort.data_sort()
            
    log.shutdown()