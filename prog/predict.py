import os, json, pickle, sys, time
import pandas as pd
from traceback import format_exc
from log_config import Log
from tools import *
from db import Database
from tqdm import tqdm



class Predict():
    def __init__(self, root, input_, logging):
        self.logging = logging
        self.model_id = input_["model_id"]
        self.start_time = input_["start_time"]
        self.end_time = input_["end_time"]

        # 取得predict位置
        pred_path = os.path.join(root, "data", "predict")        
        os.makedirs(pred_path, exist_ok = True)
        self.output_json = os.path.join(pred_path, "output.json")
        self.data_csv = os.path.join(pred_path, "data.csv")
        self.data_rm_csv = os.path.join(pred_path, "data_rm.csv")
        self.waste_json = os.path.join(pred_path, "waste.json")


        # 取得model位置     
        self.model_detail = os.path.join(root, "data", "train", self.model_id, "model")


        # 取得config        
        config_path = os.path.join(root, "prog", "config.json")
        self.logging.info(f'Read config from {config_path}')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        

        self.col_dict = {
            0: ["color", "cabinet", "length", "width"], # 同plate，同cabinet，同item
            1: ["color", "cabinet"], # 同plate，同cabinet，不同item
            2: ["color", "length", "width"], # 同plate，不同cabinet，同item
            3: ["color"], # 同plate，不同cabinet，不同item
        }



    def get_data_from_db(self):
        db_con = self.config["database"]
        self.db = Database(db_con)

        weights_df = self.db.get_data(db_con["table"]["weight"])
        self.weights = weights_df.iloc[-1][["e_ship_date", "length", "width", "area"]].astype(float).to_list()
        # self.weights = [0, 0, 0, 1]

        df = self.db.get_data(db_con["table"]["predict"], self.start_time, self.end_time)

        if df.empty:
            raise NoDataFoundException
        
        remove = (df[["length", "width"]] <= 0).any(axis = 1)
        df_rm = df[remove].reset_index(drop = True)
        df = df[~remove].reset_index(drop = True) # 長寬為0無法切割
        df = df[["order_id", "cabinet", "item_name", "color", "length", "width", "e_ship_date"]]


        return df, df_rm



    def load_model(self):
        self.features         = pickle.load(open(os.path.join(self.model_detail, "feat_order.pkl"), "rb"))
        self.outlier_boundary = pickle.load(open(os.path.join(self.model_detail, "outlier_boundary.pkl"), "rb"))
        self.skew_feat        = pickle.load(open(os.path.join(self.model_detail, "skew_feat.pkl"), "rb"))
        self.pt               = pickle.load(open(os.path.join(self.model_detail, "power_tf.pkl"), "rb"))
        self.scaler           = pickle.load(open(os.path.join(self.model_detail, "scaler.pkl"), "rb"))
        self.model            = pickle.load(open(os.path.join(self.model_detail, "model.pkl"), "rb"))
    
    

    def preprocessing(self, df, df_rm):
        df.insert(0, "selected", 0)
        df.insert(1, "label", None)
        

        self.logging.info("- Calculate area.")
        df["area"] = df["length"] * df["width"] # 每個矩形的面積
        self.length_limit = self.config["limit"]["length"]
        self.width_limit = self.config["limit"]["width"]
        area_limit = self.length_limit * self.width_limit
        df["area_prob"] = df["area"] / area_limit # 矩形站箱子的面積


        self.logging.info("- Delete Unsortable items.")
        df_rm1 = df.query("(length > @self.length_limit) or (width > @self.width_limit)")
        df_rm1 = df_rm1.drop(['selected', 'label', 'area', 'area_prob'], axis = 1)
        df_rm = pd.concat([df_rm, df_rm1], ignore_index = True)

        df = df.query("(length <= @self.length_limit) or (width <= @self.width_limit)").reset_index(drop = True)


        self.logging.info("- Initialize sort.")
        df = df.sort_values(["color", "order_id", "cabinet", "length", "width"], ascending = [True, True, True, False, False])
        df = df.reset_index(drop = True)


        self.logging.info("- Calculate the total number of data of this type.")
        # 計算第n個類型的產品的總數量
        df, self.color_accum_counts = calculate_num(col = ["color"], df = df, target = "color", flag = "predict")
        df, self.cabinet_accum_counts = calculate_num(col = ["cabinet"], df = df, target = "cabinet", flag = "predict")
        df, self.color_cabinet_accum_counts = calculate_num(col = ["color", "cabinet"], df = df, target = "color_cabinet", flag = "predict")
        df, self.item_accum_counts = calculate_num(col = ["color", "cabinet", "length", "width"], df = df, target = "item", flag = "predict")
        df, self.color_item_accum_counts = calculate_num(col = ["color", "length", "width"], df = df, target = "color_item", flag = "predict")

        df, self.color_accum_counts1 = calculate_num(col = ["order_id", "color"], df = df, target = "color1", flag = "predict")
        df, self.cabinet_accum_counts1 = calculate_num(col = ["order_id", "cabinet"], df = df, target = "cabinet1", flag = "predict")
        df, self.color_cabinet_accum_counts1 = calculate_num(col = ["order_id", "color", "cabinet"], df = df, target = "color_cabinet1", flag = "predict")
        df, self.item_accum_counts1 = calculate_num(col = ["order_id", "color", "cabinet", "length", "width"], df = df, target = "item1", flag = "predict")
        df, self.color_item_accum_counts1 = calculate_num(col = ["order_id", "color", "length", "width"], df = df, target = "color_item1", flag = "predict")


        return df, df_rm
    


    def generate_feat(self, df, index):
        # 抓出產品的類型        
        color_key = df.loc[index, "color"]
        cabinet_key = df.loc[index, "cabinet"]
        color_cabinet_key = tuple([i for i in df.loc[index, ["color", "cabinet"]]])
        item_key = tuple([i for i in df.loc[index, ["color", "cabinet", "length", "width"]]])    
        color_item_key = tuple([i for i in df.loc[index, ["color", "length", "width"]]])

        color_key1 = tuple([i for i in df.loc[index, ["order_id", "color"]]])
        cabinet_key1 = tuple([i for i in df.loc[index, ["order_id", "cabinet"]]])
        color_cabinet_key1 = tuple([i for i in df.loc[index, ["order_id", "color", "cabinet"]]])
        item_key1 = tuple([i for i in df.loc[index, ["order_id", "color", "cabinet", "length", "width"]]])    
        color_item_key1 = tuple([i for i in df.loc[index, ["order_id", "color", "length", "width"]]])

        # 產品的類型的目前數量加1
        self.color_accum_counts[color_key] += 1
        self.cabinet_accum_counts[cabinet_key] += 1
        self.color_cabinet_accum_counts[color_cabinet_key] += 1
        self.item_accum_counts[item_key] += 1
        self.color_item_accum_counts[color_item_key] += 1
        
        self.color_accum_counts1[color_key1] += 1
        self.cabinet_accum_counts1[cabinet_key1] += 1
        self.color_cabinet_accum_counts1[color_cabinet_key1] += 1
        self.item_accum_counts1[item_key1] += 1
        self.color_item_accum_counts1[color_item_key1] += 1

        # 計算第n個類型的產品佔該類型產品總數量的比值
        df.loc[index, "color_num"] = self.color_accum_counts[color_key] / df.loc[index, "color_num"]
        df.loc[index, "cabinet_num"] = self.cabinet_accum_counts[cabinet_key] / df.loc[index, "cabinet_num"]
        df.loc[index, "color_cabinet_num"] = self.color_cabinet_accum_counts[color_cabinet_key] / df.loc[index, "color_cabinet_num"]
        df.loc[index, "item_num"] = self.item_accum_counts[item_key] / df.loc[index, "item_num"]
        df.loc[index, "color_item_num"] = self.color_item_accum_counts[color_item_key] / df.loc[index, "color_item_num"]
        
        df.loc[index, "color1_num"] = self.color_accum_counts1[color_key1] / df.loc[index, "color1_num"]
        df.loc[index, "cabinet1_num"] = self.cabinet_accum_counts1[cabinet_key1] / df.loc[index, "cabinet1_num"]
        df.loc[index, "color_cabinet1_num"] = self.color_cabinet_accum_counts1[color_cabinet_key1] / df.loc[index, "color_cabinet1_num"]
        df.loc[index, "item1_num"] = self.item_accum_counts1[item_key1] / df.loc[index, "item1_num"]
        df.loc[index, "color_item1_num"] = self.color_item_accum_counts1[color_item_key1] / df.loc[index, "color_item1_num"]


        return df
    


    def get_target(self, df, index):
        x_test = df.loc[[index], self.features].copy()

        # remove outlier 
        for col in self.features:
            min_ = self.outlier_boundary[col]["min"]
            max_ = self.outlier_boundary[col]["max"]
            x_test[col] = x_test[col].apply(lambda X: max_ if X > max_ else X)
            x_test[col] = x_test[col].apply(lambda X: min_ if X < min_ else X)        
        
        x_test[self.skew_feat] = self.pt.transform(x_test[self.skew_feat]) # skewing
        x_test[self.features] = self.scaler.transform(x_test[self.features]) # scaling
        df.loc[index, "label"] = self.model.predict(x_test)[0] # predict


        return df

    
    
    def predict(self, progress_gap = 1):
        try:
            self.logging.info("Get data from database.")
            df, df_rm = self.get_data_from_db()


            self.logging.info(f'Load model from {self.model_detail}\*.pkl')
            self.load_model()


            self.logging.info("Preprocessing.")
            df, df_rm = self.preprocessing(df, df_rm)
            
            self.logging.info("Predict and sort.")
            df[["accum_areas", "waste", "plate_id", "fail"]] = [0, 1, 0, 0]

            # 初始化第0個板材
            df.loc[0, "plate_id"] = 1
            df.loc[0, "accum_areas"] = df.loc[0, "area"] # 箱中的累積使用面積
            df.loc[0, "waste"] = (1 - (df.loc[0, "accum_areas"] / (self.length_limit * self.width_limit))) # 計算waste欄位，及耗損率

            prev = 0
            pbar = tqdm(total = len(df)-2, ncols = 150)

            # 依序選出板材
            for index in range(len(df)-1):
                df.loc[index, "selected"] = 1 # 選到的板材的selected欄位由0改為1


                ## 計算第n個類型的產品佔該類型產品總數量的比值，ex: color_num、cabinet_num、color_cabinet_num、item_num欄位
                df = self.generate_feat(df, index)

                ## 預測target
                df = self.get_target(df, index)

                ## 抓出下一塊板材的index
                df = get_next_plate(self.col_dict, index, self.weights, df, self.length_limit, self.width_limit)

                ## 更新前端進度條
                now = time.time()
                if (now - prev) > progress_gap:
                    percent = round((0.9 / len(df)) * (index + 1), 2)
                    self.db.save_progress(percent = percent)                    
                    prev = now
                
                
                pbar.set_postfix({"plate_id": df.loc[index+1, "plate_id"], "waste": df.loc[index+1, "waste"]})
                pbar.update(1)
                
            pbar.close()

            self.logging.info(f"Save data to {self.data_csv}")
            df.insert(0, "order", df.index)
            keep_col = ['order', 'order_id', 'cabinet', 'item_name', 'color', 'length', 'width', 'area', 'area_prob', 'plate_id', 'accum_areas', 'waste', "e_ship_date"]
            df1 = df[keep_col].round(4)
            df1.to_csv(self.data_csv, encoding='utf-8-sig', index = False)
            df_rm.to_csv(self.data_rm_csv, encoding='utf-8-sig', index = False)

            
            self.logging.info(f"Save waste to {self.waste_json}")
            calculate_mean_wast(df, self.waste_json) # 耗損率


            result = {
                "status": "success",
                "model_id": self.model_id, 
                "data_counts": len(df1),
                "remove_counts": len(df_rm),
                }


        except (pd.errors.EmptyDataError, NoDataFoundException):
            message = "No data is available in the database for the specified date range."
            result = error(self.logging, message, self.model_id)
        
        
        except:
            message = format_exc()
            result = error(self.logging, message, self.model_id)


        finally:
            self.logging.info(f'Save output to {self.output_json}')
            with open(self.output_json, 'w') as file:
                json.dump(result, file, indent = 4)
            
            self.db.save_progress(percent = 1)



class NoDataFoundException(Exception):
    pass



if __name__ == '__main__':
    # 取得根目錄
    current_path = os.path.abspath(__file__)
    prog_path = os.path.dirname(current_path)
    root = os.path.dirname(prog_path)


    log = Log()
    log_path = os.path.join(root, "logs")
    os.makedirs(log_path, exist_ok = True)
    logging = log.set_log(filepath = os.path.join(log_path, "predict.log"), level = 2, freq = "D", interval = 50, backup = 3, name = "predict")
    
    logging.info("-"*200)
    # logging.info(f"root: {root}")
    

    input_ = get_input(sys.argv, logging)
    logging.info(f"input = {input_}")


    model = Predict(root, input_, logging)
    model.predict()

    
    log.shutdown()