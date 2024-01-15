# ZA XGboost

pip install xgboost

import xgboost as xgb

model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

for mask in masks:          # meni je ovo iteracija po withdrawn, fail, pass i tim klasama za predikciju
    mask = mask[-5000:]
    Y = mask.astype(int)

    #encoded_df su mi sredeni podaci bez null vrijednosti
    X_train, X_test, y_train, y_test = train_test_split(encoded_df, Y, test_size=0.3)

    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = lambda x : model.predict(x) >= 0.5

    pred = y_pred(X_test)
    pred = [int(x) for x in pred]

    print('y_pred: ', pred)

    accuracy = metrics.accuracy_score(y_test, pred)

    print('accuracy: ', accuracy)


## KRAJ XGboosta
    

## POKUSAJ SA DECISION TREE I UNAKRSNOM PROVJEROM
    from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler


# OVAJ PRVI DIO JE UCITAVANJE MASKA I KODIRANJE PODATAKA:
mask_pass = mask_pass = merged.final_result == 'Pass'
mask_fail = merged.final_result == 'Fail'
mask_distinction = merged.final_result == 'Distinction'
mask_withdrawn = merged.final_result == 'Withdrawn'


label_encoder = LabelEncoder()


#columns_to_keep = ['gender',	'age_band',	'highest_education',	'score'	,'date_registration',	'sum_click']
columns_to_keep = ['highest_education', 'age_band', 'disability', 'gender', 'score', 'sum_click', 'date_registration', 'num_of_prev_attempts']
merged2 = merged[columns_to_keep]

encoded_df = pd.DataFrame(merged2)
encoded_df['highest_education'] = label_encoder.fit_transform(merged2['highest_education'])
encoded_df['age_band'] = label_encoder.fit_transform(merged2['age_band'])
encoded_df['disability'] = label_encoder.fit_transform(merged2['disability'])
encoded_df['gender'] = label_encoder.fit_transform(merged2['gender'])
#print(encoded_df)
#encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['highest_education', 'age_band', 'disability', 'gender']))
encoded_df['score'] = merged2['score'].values
encoded_df['sum_click'] = merged2['sum_click'].values
encoded_df['date_registration'] = merged2['date_registration'].values
encoded_df['num_of_prev_attempts'] = merged2['num_of_prev_attempts'].values

encoded_df = encoded_df.dropna()

print(len(encoded_df))
encoded_df = encoded_df[-5000:]
#print('encoded_score: ', encoded_df.columns[encoded_df.isnull().any()].tolist())


masks = [mask_pass, mask_fail, mask_distinction, mask_withdrawn]
final_result = ['Withdrawn', 'Pass', 'Fail', 'Distinction']
## KRAJ SREDIVANJA PODATKA I MASKI

i = 0

decision_tree_rez = []
for mask in masks:
    rez = []
    mask = mask[-5000:]
    Y = mask.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(encoded_df, Y, test_size=0.2, random_state=42)

    # za poboljsanje kod:
    param_grid = {
    'max_depth': [3, 5, 7, 9],
    'min_samples_split': [3, 7, 5, 9],
    'min_samples_leaf': [1, 3, 5, 7],
    'max_features': ['sqrt', 'log2', None]
    }

    scaler = StandardScaler()
    X_train_standardized = scaler.fit_transform(X_train)

    X_test_standardized = scaler.transform(X_test)

    grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    grid_search.fit(X_train_standardized, y_train)

    best_params = grid_search.best_params_

    print('best_params: ', best_params)

    rez = []

    optimized_model = DecisionTreeClassifier(**best_params)

    optimized_model.fit(X_train, y_train)


    y_pred = cross_val_predict(optimized_model, X_test_standardized, y_test, cv=5)



    accuracy = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy for decision tree 1 for ' + final_result[i] + ': ' + str(accuracy))

    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print('Precision for decision tree1 for ' + final_result[i] + ': ' + str(precision))
    print('Recall for decision tree1 for ' + final_result[i] + ': ' + str(recall))
    print('F1 for decision tree1 for ' + final_result[i] + ': ' + str(f1_score))

    ''' rez = [final_result[i]]
    rez.append ('Precision')
    rez.append(str(round(precision, 2)))

    decision_tree_rez.append(rez)

    rez = [final_result[i]]
    rez.append ('Recall')
    rez.append(str(round(recall, 2)))
    decision_tree_rez.append(rez)

    rez = [final_result[i]]
    rez.append ('F1')
    rez.append(str(round(f1_score, 2)))
    decision_tree_rez.append(rez)

    rez = [final_result[i]]
    rez.append ('Accuracy')
    rez.append(str(round(accuracy, 2)))  
    decision_tree_rez.append(rez) '''
      

    clf = DecisionTreeClassifier()

    clf.fit(X_train, y_train)

    #y_pred = clf.predict(X_test)

    y_pred = cross_val_predict(clf, X_test, y_test, cv=5)


    accuracy = metrics.accuracy_score(y_test, y_pred)
    print('Accuracy for decision tree 2 for ' + final_result[i] + ': ' + str(accuracy))

    precision, recall, f1_score, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print('Precision for decision tree 2 for ' + final_result[i] + ': ' + str(precision))
    print('Recall for decision tree 2 for ' + final_result[i] + ': ' + str(recall))
    print('F1 for decision tree 2 for ' + final_result[i] + ': ' + str(f1_score))

    i += 1

#print('dt: ', decision_tree_rez)

## KRAJ UNAKRSNE PROVJERE ZA DECISION TREE
    

## SVM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

i= 0
for mask in masks:
        rez = []

        mask = mask[-5000:]
        Y = mask.astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(encoded_df, Y, test_size=0.3, random_state=42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        svm_classifier = SVC(kernel='linear', C=1.0)  # Linear kernel for simplicity, C is the regularization parameter

        svm_classifier.fit(X_train, y_train)

        y_pred = svm_classifier.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print('Accuracy for SVM for ' + final_result[i] + ': ' + str(accuracy))

        i += 1

## KRAJ SVM-A
    


