from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import folium
from folium import plugins

# Matplotlib backend'ini 'Agg' olarak ayarlama
matplotlib.use('Agg')

app = Flask(__name__, static_url_path='/static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static/uploads'

# Ensure the static/uploads directory exists
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('secim', filename=file.filename))
    return render_template('index.html')

@app.route('/secim/<filename>')
def secim(filename):
    return render_template('secim.html', filename=filename)

@app.route('/results/<filename>')
def results(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_csv(filepath)
    
    # 'id' sütununu veri setinden çıkarma
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    
    # Eksik değerleri olan satırları çıkarma
    df = df.dropna()
    
    # Güncellenmiş veriyi kaydetme
    df.to_csv(filepath, index=False)
    
    # Sayısal ve kategorik sütunların isimlerini ve sayılarını belirleme
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_numeric_cols = len(numeric_cols)
    num_categorical_cols = len(categorical_cols)
    
    # Sayısal sütunlar için istatistiksel hesaplamalar
    numeric_stats = df[numeric_cols].describe().to_html(classes='data', header="true", index=True)
    
    # Sayısal sütunlar için ilk 5 satırı gösterme
    numeric_head = df[numeric_cols].head().to_html(classes='data', header="true", index=False)
    
    # Kategorik sütunlar için ilk 5 satırı gösterme (varsa)
    categorical_head = None
    if num_categorical_cols > 0:
        categorical_head = df[categorical_cols].head().to_html(classes='data', header="true", index=False)
    
    # Verinin ilk 5 satırını gösterme
    full_head = df.head().to_html(classes='data', header="true", index=False)

    # Aykırı değer analizi
    outlier_counts = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_counts[col] = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()

    # Aykırı değerleri gösteren tabloyu oluşturma
    outliers_table = pd.DataFrame(list(outlier_counts.items()), columns=['Column', 'Outlier Count']).to_html(classes='data', header="true", index=False)

    # Duplike olan satırları bulma
    duplicated_rows = df[df.duplicated()]
    duplicated_table = None
    if not duplicated_rows.empty:
        duplicated_table = duplicated_rows.to_html(classes='data', header="true", index=False)
    else:
        duplicated_table = "<p>There are no duplicated rows in the dataframe!</p>"

    # Kategorik sütunların benzersiz değer sayılarını gösteren grafik (varsa)
    categorical_unique_path = None
    if num_categorical_cols > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        df[categorical_cols].nunique().plot(kind='bar', ax=ax)
        ax.set_title('Unique Values in Categorical Columns')
        categorical_unique_path = os.path.join(app.config['STATIC_FOLDER'], 'categorical_unique.png')
        plt.savefig(categorical_unique_path)
        plt.close()

    # Sayısal sütunlar için box plot (varsa)
    numeric_boxplot_path = None
    if num_numeric_cols > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df[numeric_cols], ax=ax)
        ax.set_title('Box Plot of Numerical Columns')
        numeric_boxplot_path = os.path.join(app.config['STATIC_FOLDER'], 'numeric_boxplot.png')
        plt.savefig(numeric_boxplot_path)
        plt.close()

    # Sadece sayısal değişkenleri seçiyoruz
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    # Hedef değişkeni ayırma ve kategorik hale getirme
    if 'mw' in df.columns:
        bins = [-float('inf'), 3.0, 4.0, 5.0, float('inf')]
        labels = ['Genellikle hissedilmez, risksiz', 'Küçük, genellikle hissedilmez, düşük riskli', 'Hafif, hissedilir, düşük ila orta riskli', 'Orta büyüklükte, hissedilir, orta riskli']
        df['Mw_category'] = pd.cut(df['mw'], bins=bins, labels=labels)
        X = df[numerical_cols].drop('mw', axis=1)
        y = df['Mw_category']
    else:
        X = df[numerical_cols]
        y = None

    # Feature scaling işlemi
    robust_scaler = RobustScaler()
    X_scaled = robust_scaler.fit_transform(X)

    # Veriyi eğitim ve test setlerine bölme
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # RandomForestClassifier modelini eğit
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Modelin performansını değerlendir
    accuracy = model.score(X_test, y_test)

    # Çapraz doğrulama ile modelin performansını değerlendir
    cross_val_scores = cross_val_score(model, X_scaled, y, cv=5)
    cross_val_mean = cross_val_scores.mean()

    # Karışıklık matrisi
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
    cm_display_path = os.path.join(app.config['STATIC_FOLDER'], 'confusion_matrix.png')
    cm_display.plot()
    plt.title('Confusion Matrix')
    plt.savefig(cm_display_path)
    plt.close()

    # Hedef değişkenin dağılımını gösteren pasta grafiği oluşturma
    counts = df['Mw_category'].value_counts().to_dict()
    total = sum(counts.values())

    fig, ax = plt.subplots()
    ax.pie(counts.values(), labels=counts.keys(), autopct=lambda pct: '{:.1f}% ({:d})'.format(pct, int(pct*total/100)))
    ax.set_title("Deprem Büyüklüğü Dağılımı")
    pie_chart_path = os.path.join(app.config['STATIC_FOLDER'], 'pie_chart.png')
    plt.savefig(pie_chart_path)
    plt.close()
    
    # Folium haritasını oluşturma ve kaydetme
    m = folium.Map(location=[df["lat"].mean(), df["long"].mean()], zoom_start=3)
    heat_data = df[["lat", "long"]].dropna().values
    m.add_child(plugins.HeatMap(heat_data))
    map_path = os.path.join(app.config['STATIC_FOLDER'], 'map.html')
    m.save(map_path)

    return render_template(
        'results.html', 
        numeric_head=numeric_head, 
        numeric_stats=numeric_stats, 
        categorical_head=categorical_head,
        full_head=full_head,
        outliers_table=outliers_table,
        duplicated_table=duplicated_table,
        num_numeric_cols=num_numeric_cols,
        num_categorical_cols=num_categorical_cols,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        categorical_unique_path=url_for('static', filename='uploads/categorical_unique.png'),
        numeric_boxplot_path=url_for('static', filename='uploads/numeric_boxplot.png'),
        pie_chart_path=url_for('static', filename='uploads/pie_chart.png'),
        cm_display_path=url_for('static', filename='uploads/confusion_matrix.png'),
        accuracy=accuracy,
        cross_val_mean=cross_val_mean
    )

@app.route('/harita/<filename>')
def harita(filename):
    # Harita dosyasının doğru şekilde oluşturulduğundan emin olun
    map_path=url_for('static', filename='uploads/map.html')
    return render_template('harita.html', map_path=map_path)

if __name__ == '__main__':
    app.run(debug=True)




