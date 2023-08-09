import os, json, pickle, sys
import pandas as pd
from traceback import format_exc
from log_config import Log
from tools import *
from db import Database
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, ConfusionMatrixDisplay, confusion_matrix, classification_report
import matplotlib.pyplot as plt



class Model():
    def __init__(self, root, input_, logging):
        self.logging = logging

        self.model_id   = input_["model_id"]
        self.start_time = input_["start_time"]
        self.end_time   = input_["end_time"]


        # 取得train位置
        train_path = os.path.join(root, "data", "train")        
        os.makedirs(train_path, exist_ok = True)
        self.output_json = os.path.join(train_path, "output.json")


        # 取得model位置
        self.model_path = os.path.join(root, "data", "train", self.model_id)
        os.makedirs(self.model_path, exist_ok = True)
        
        self.model_detail = os.path.join(self.model_path, "model")
        os.makedirs(self.model_detail, exist_ok = True)


        # 取得config        
        config_path = os.path.join(root, "prog", "config.json")
        self.logging.info(f'Read config from {config_path}')
        with open(config_path, 'r') as f:
            self.config = json.load(f)



    def get_data_from_db(self):
            db_con  = self.config["database"]
            self.db = Database(db_con)

            df = self.db.get_data(db_con["table"]["train"], self.start_time, self.end_time)

            if df.empty:
                raise NoDataFoundException
            
            df = df.sort_values("serial").reset_index(drop = True)
            df = df[~(df[["length", "width"]] <= 0).any(axis = 1)].reset_index() # 長寬為0無法切割
            df = df[["order_id", "cabinet", "item_name", "color", "length", "width", "e_ship_date"]]


            return df
    

    
    def preprocessing(self, df, target):
        self.logging.info("- Calculate area.")
        df["area"]      = df["length"] * df["width"] # 每個矩形的面積
        length_limit    = self.config["limit"]["length"]
        width_limit     = self.config["limit"]["width"]
        area_limit      = length_limit * width_limit
        df["area_prob"] = df["area"] / area_limit # 矩形站箱子的面積


        self.logging.info("- Calculate waste and plate_id.")
        df = calculate_waste(df, length_limit, width_limit)


        self.logging.info("- Calculate the nth data of this type divided by the total number of data of this type.")
        # 計算第n個類型的產品佔該類型產品總數量的比值
        df = calculate_num(col = ["color"], df = df, target = "color", flag = "train")
        df = calculate_num(col = ["cabinet"], df = df, target = "cabinet", flag = "train")
        df = calculate_num(col = ["color", "cabinet"], df = df, target = "color_cabinet", flag = "train")
        df = calculate_num(col = ["color", "length", "width"], df = df, target = "color_item", flag = "train")
        df = calculate_num(col = ["color", "cabinet", "length", "width"], df = df, target = "item", flag = "train")
        
        df = calculate_num(col = ["order_id", "color"], df = df, target = "color1", flag = "train")
        df = calculate_num(col = ["order_id", "cabinet"], df = df, target = "cabinet1", flag = "train")
        df = calculate_num(col = ["order_id", "color", "cabinet"], df = df, target = "color_cabinet1", flag = "train")
        df = calculate_num(col = ["order_id", "color", "length", "width"], df = df, target = "color_item1", flag = "train")
        df = calculate_num(col = ["order_id", "color", "cabinet", "length", "width"], df = df, target = "item1", flag = "train")
        

        self.logging.info("- Generate label.")
        df = generate_label(df, target)
        df = df.drop(["order_id", "cabinet", "item_name", "color", "e_ship_date"], axis = 1)


        return df



    def feature_engineering(self, X_train, X_test):
        self.logging.info("- Deal with outlier.")
        outlier_boundary = {}
        for col in self.features:
            Q1   = X_train[col].quantile(0.25)
            Q3   = X_train[col].quantile(0.75)
            IQR  = Q3 - Q1
            min_ = Q1 - (1.5 * IQR)
            max_ = Q3 + (1.5 * IQR)
            
            X_train[col] = X_train[col].apply(lambda X: max_ if X > max_ else X)
            X_train[col] = X_train[col].apply(lambda X: min_ if X < min_ else X)

            X_test[col]  = X_test[col].apply(lambda X: max_ if X > max_ else X)
            X_test[col]  = X_test[col].apply(lambda X: min_ if X < min_ else X)

            outlier_boundary[col] = {
                "min": min_,
                "max": max_,
            }
        self.outlier_boundary = outlier_boundary


        self.logging.info("- Deal with skew.")
        skewness = X_train[self.features].apply(lambda X: skew(X)).sort_values(ascending=False)
        skewness = pd.DataFrame({'Feature' : skewness.index, 'Skew' : skewness.values})
        skewness = skewness.query("(Skew > 0.75) | (Skew < -0.75)")
        skewness = skewness.reset_index(drop = True)
        self.skew_feat = skewness["Feature"].to_list()


        pt = PowerTransformer(method = 'yeo-johnson')
        X_train[self.skew_feat] = pt.fit_transform(X_train[self.skew_feat])
        X_test[self.skew_feat]  = pt.transform(X_test[self.skew_feat])
        self.pt = pt

        
        self.logging.info("- Scaling.")
        scaler = StandardScaler()
        X_train[self.features] = scaler.fit_transform(X_train[self.features])
        X_test[self.features]  = scaler.transform(X_test[self.features])
        self.scaler = scaler
        

        return X_train, X_test
    
    

    def calculate_score(self, pred_train, pred_test, y_train, y_test):
        acc_train  = accuracy_score(y_train, pred_train).round(2)
        acc_test   = accuracy_score(y_test, pred_test).round(2)

        recall_train  = recall_score(y_train, pred_train, average = 'weighted').round(2)
        recall_test   = recall_score(y_test, pred_test, average = 'weighted').round(2)

        precision_train  = precision_score(y_train, pred_train, average = 'weighted').round(2)
        precision_test   = precision_score(y_test, pred_test, average = 'weighted').round(2)

        f1_train = f1_score(y_train, pred_train, average = 'weighted').round(2)
        f1_test  = f1_score(y_test, pred_test, average = 'weighted').round(2)

        score = {
            "length": {
                "train": len(y_train), "test": len(y_test),
            },
            "accuracy": {
                "train": acc_train, "test": acc_test,
            },
            "precision": {
                "train": precision_train, "test": precision_test
            },
            "recall": {
                "train": recall_train, "test": recall_test
            },
            "f1": {
                "train": f1_train, "test": f1_test,
            },
        }
        
        
        # report = classification_report(y_test, pred_test)
        # self.logging.info(f"Save score.\n{report}")

        score = pd.DataFrame(score)
        score.to_csv(os.path.join(self.model_path, "score.csv"))
        
        
        return score["accuracy"]["test"]

    
    
    def save_chart(self, pred_train, pred_test, y_train, y_test):
        # confusion matrix
        fig, ax = plt.subplots(1, 2, figsize = (10, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_train, pred_train), display_labels = (set(pred_train) | set(y_train)))
        disp.plot(cmap = plt.cm.Blues, ax = ax[0])
        ax[0].set_title(f"train - confusion matrix")

        disp = ConfusionMatrixDisplay(confusion_matrix = confusion_matrix(y_test, pred_test), display_labels = (set(pred_test) | set(y_test)))
        disp.plot(cmap = plt.cm.Blues, ax = ax[1])
        ax[1].set_title(f"test - confusion matrix")
        fig.tight_layout()
        fig.savefig(os.path.join(self.model_path, "confusion_matrix.png"))


        # Feature importance
        fig, ax = plt.subplots(figsize = (15, 5))
        imp = pd.DataFrame(self.model.feature_importances_, index = self.features, columns = ["importance"])
        imp = imp.query("importance != 0").sort_values("importance")
        imp.plot(kind = "barh", ax = ax, fontsize = 12)
        ax.set_title("Feature importance", fontsize = 14)
        fig.savefig(os.path.join(self.model_path, "feature_importance.png"))



    def save_model(self):
        pickle.dump(self.features, open(os.path.join(self.model_detail, "feat_order.pkl"), "wb"))
        pickle.dump(self.outlier_boundary, open(os.path.join(self.model_detail, "outlier_boundary.pkl"), "wb"))
        pickle.dump(self.skew_feat, open(os.path.join(self.model_detail, "skew_feat.pkl"), "wb"))
        pickle.dump(self.pt, open(os.path.join(self.model_detail, "power_tf.pkl"), "wb"))
        pickle.dump(self.scaler, open(os.path.join(self.model_detail, "scaler.pkl"), "wb"))
        pickle.dump(self.model, open(os.path.join(self.model_detail, "model.pkl"), "wb"))



    def train(self, target = "label", random_state = 99):
        try:
            self.logging.info("Get data from database.")
            df = self.get_data_from_db()
            

            self.logging.info("Preprocessing.")
            df = self.preprocessing(df, target)


            self.logging.info("Feature engineering.")
            self.features = list(df.columns[1:])
            X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis = 1), df[target], test_size=0.2, shuffle = False)

            X_train, X_test = self.feature_engineering(X_train, X_test)


            self.logging.info("Modeling.")
            xgb_args = {'subsample': 0.6, 'n_estimators': 150, 'min_child_weight': 5, 'max_depth': 5, 'learning_rate': 0.05, 'gamma': 0.4, 'colsample_bytree': 0.7}
            self.model = XGBClassifier(random_state = random_state, **xgb_args)
            self.model.fit(X_train, y_train)

            pred_train = self.model.predict(X_train)
            pred_test  = self.model.predict(X_test)

            
            self.logging.info(f'Save score and chart.')
            accuracy = self.calculate_score(pred_train, pred_test, y_train, y_test)
            self.save_chart(pred_train, pred_test, y_train, y_test)

            
            self.logging.info(f'Save model to {self.model_detail}\*.pkl')
            self.save_model()

            result = {
                "status":   "success",
                "reason":   "",
                "model_id": self.model_id,
                "accuracy": accuracy,
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
    


class NoDataFoundException(Exception):
    pass



if __name__ == '__main__':
    # 取得根目錄
    current_path = os.path.abspath(__file__)
    prog_path    = os.path.dirname(current_path)
    root = os.path.dirname(prog_path)


    log = Log()
    log_path = os.path.join(root, "logs")
    os.makedirs(log_path, exist_ok = True)
    logging = log.set_log(filepath = os.path.join(log_path, "train.log"), level = 2, freq = "D", interval = 50, backup = 3, name = "train")
    
    logging.info("-"*200)
    # logging.info(f"root: {root}")
    

    input_ = get_input(sys.argv, logging)
    logging.info(f"input = {input_}")


    model = Model(root, input_, logging)
    model.train()
            
    log.shutdown()