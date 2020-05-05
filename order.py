import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.linear_model import LogisticRegression as LREG
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.multiclass import unique_labels

pd.set_option('display.max_columns', None)


customers = pd.read_csv('brazilian-ecommerce/olist_customers_dataset.csv')

orders = pd.read_csv('brazilian-ecommerce/olist_orders_dataset.csv')

order_reviews = pd.read_csv('brazilian-ecommerce/olist_order_reviews_dataset.csv')

order_items = pd.read_csv('brazilian-ecommerce/olist_order_items_dataset.csv')

order_payments = pd.read_csv('brazilian-ecommerce/olist_order_payments_dataset.csv')

products = pd.read_csv('brazilian-ecommerce/olist_products_dataset.csv')

sellers = pd.read_csv('brazilian-ecommerce/olist_sellers_dataset.csv')

geolocation = pd.read_csv('brazilian-ecommerce/olist_geolocation_dataset.csv')



def missing_values_table(df):
    # Total # of missing values
    mis_val = df.isnull().sum()

    # Proportion of missing values
    mis_val_percent = 100 * mis_val / len(df)

    # Create a table containing the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Set column names
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Number of Missing Values', 1: 'Percentage of Total Values (%)'})

    # Sort the table by percentage of missing values (descending order)
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        'Percentage of Total Values (%)', ascending=False).round(3)

    # Print summary information
    print("The dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns having missing values.")

    # Results output
    return mis_val_table_ren_columns

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalise=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          multi=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalise=True`.
    """
    if not title:
        if normalise:
            title = 'Normalised confusion matrix'
        else:
            title = 'Confusion matrix, without normalisation'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Only use the labels that appear in the data
    if multi == True:
        classes = classes[unique_labels(y_true, y_pred)]
    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor");

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax

# cleaning orders
orders = orders.drop(['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_status'],axis=1)
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'], format='%Y/%m/%d %H:%M:%S')
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'], format='%Y/%m/%d %H:%M:%S')
orders['time_difference'] = (orders['order_estimated_delivery_date'] - orders['order_delivered_customer_date']).dt.total_seconds()
orders = orders.drop(['order_delivered_customer_date','order_estimated_delivery_date'],axis=1)
# orders = pd.get_dummies(orders, columns=['order_status'], prefix=['order_status'])
orders = orders[~(orders['time_difference'].isnull())]
# print(orders)

# cleaning order_reviews
order_reviews = order_reviews.drop(['review_creation_date','review_answer_timestamp'],axis=1)
order_reviews['review_comment_title']
order_reviews['isComment'] = order_reviews['review_comment_title'].notnull()*1 + order_reviews['review_comment_message'].notnull()*1
order_reviews['isComment'] = order_reviews['isComment'].apply(lambda x: 1 if x>=1  else 0)
order_reviews = order_reviews.drop(['review_comment_title','review_comment_message'],axis=1)
order_reviews['review_score'] = order_reviews['review_score'].apply(lambda x: 1 if x>=4  else 0)

# print(order_reviews)

# cleaning order_items
order_items = order_items.drop(['shipping_limit_date'],axis=1)
# print(order_items)


# cleaning order_payments
order_payments = order_payments.drop(['payment_sequential','payment_value'],axis=1)
order_payments = pd.get_dummies(order_payments, columns=['payment_type'], prefix=['payment_type'])
order_payments['payment_installments'] = order_payments['payment_installments'].apply(lambda x: 0 if x<=1  else 1)
# print(order_payments)


# cleaning products
products = products.drop(['product_name_lenght','product_weight_g','product_length_cm','product_height_cm','product_width_cm'],axis=1)
products = products[~(products['product_category_name'].isnull())]
products = pd.get_dummies(products, columns=['product_category_name'], prefix=['products_category'])
# print(products)


# cleaning sellers
# print(sellers)

# cleaning geolocation
# print(geolocation)


orders = orders.merge(customers,on='customer_id')
orders = orders.merge(order_reviews,on='order_id')
orders = orders.merge(order_payments,on='order_id')
# print(orders)

order_items = order_items.merge(products,on='product_id')
order_items = order_items.merge(sellers,on='seller_id')
# sellers = sellers.merge(geolocation,left_on='seller_zip_code_prefix',right_on='geolocation_zip_code_prefix')

orders = orders.merge(order_items,on='order_id')

orders['isSameState'] = orders.apply(lambda x: 1 if x['customer_state'] == x['seller_state'] else 0, axis = 1)
orders = orders.drop(['order_id','customer_id','customer_unique_id','customer_zip_code_prefix','customer_city','review_id','product_id','seller_id','seller_zip_code_prefix','seller_city','customer_state','seller_state'],axis=1)

# print(orders)
# print(missing_values_table(orders))

label1 = orders['isComment']
label2 = orders['review_score']
data = orders.drop(['review_score','isComment'],axis=1)

# Normalization
data = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns)

# ========================================model 1========================================
X_train, X_test, Y_train, Y_test = train_test_split(data, label1, test_size = 0.1, random_state=5)

# print the shapes to check everything is OK
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)

#===============================dtc start===============================
# tuned_parameters = [{'criterion': ['gini', 'entropy'],
#                      'max_depth': [3, 5, 7],
#                      'min_samples_split': [3, 5, 7],
#                      'max_features': ["sqrt", "log2", None]}]
#
# scores = ['accuracy', 'f1_macro']
# for score in scores:
#     print("# Tuning hyperparameters for %s" % score)
#     print("\n")
#     clf = GridSearchCV(DTC(), tuned_parameters, cv=5, scoring='%s' % score)
#     clf.fit(X_train, Y_train)
#     print("Best parameters set found on the training set:")
#     print(clf.best_params_)
#     print("\n")

# a decision tree model with default values
dtc = DTC(criterion='gini', min_samples_split=5, max_depth=7, max_features=None)

# fit the model using some training data
dtc_fit = dtc.fit(X_train, Y_train)

# generate a mean accuracy score for the predicted data
train_score = dtc.score(X_train, Y_train)

# print the mean accuracy of testing predictions
print("Accuracy score = " + str(round(train_score, 4)))

# predict the test data
predicted = dtc.predict(X_test)

# Plot non-normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"])

# Plot normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"], normalise=True)
#===============================dtc end===============================

# ========================================model 1========================================



# ========================================model 2========================================

X_train, X_test, Y_train, Y_test = train_test_split(data, label2, test_size = 0.1, random_state=5)

# print the shapes to check everything is OK
# print(X_train.shape)
# print(X_test.shape)
# print(Y_train.shape)
# print(Y_test.shape)
#===============================lr start===============================
# tuned_parameters = [{'penalty': ['l2','none'],
#                      'solver': ['lbfgs','newton-cg','sag'],
#                      'multi_class': ['multinomial']}]
#
# scores = ['accuracy', 'roc_auc']
#
# for score in scores:
#     print("# Tuning hyperparameters for %s" % score)
#     print("\n")
#     clf = GridSearchCV(LREG(random_state = 0), tuned_parameters, cv=5, scoring=score)
#     clf.fit(X_train, Y_train)
#     print("Best parameters set found on the training set:")
#     print(clf.best_params_)
#     print("\n")

# a logistic regression model with default values
lreg = LREG(multi_class='multinomial',penalty='l2',solver='lbfgs',random_state = 0)
# fit the model using some training data
lreg_fit = lreg.fit(X_train, Y_train)
# generate a mean accuracy score for the training data
train_score = lreg.score(X_train, Y_train)
# print the R2 of training data
print("Logistic regression R2 (Train) = " + str(round(train_score, 4)))

# predict the test data
predicted = lreg_fit.predict(X_test)

# generate a mean accuracy score for the predicted data
test_score = lreg_fit.score(X_test, Y_test)

# print the R2 of testing predictions
print("Logistic regression R2 (Test) = " + str(round(test_score, 4)))

# predict the test data
predicted = lreg.predict(X_test)

# Plot non-normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"])

# Plot normalised confusion matrix
plot_confusion_matrix(Y_test, predicted, classes=["0", "1"], normalise=True)

#===============================lr end===============================

# ========================================model 1========================================


