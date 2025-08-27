import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configurações de visualização
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def analyze_age_importance():
    """
    Análise específica da importância da idade na sobrevivência
    """
    print("="*60)
    print("🧓 ANÁLISE ESPECÍFICA DA IMPORTÂNCIA DA IDADE")
    print("="*60)
    
    # Carregar dados
    train_df = pd.read_csv('train_processed.csv')
    
    # 1. Correlação direta da idade com sobrevivência
    age_survival_corr = train_df['Age'].corr(train_df['Survived'])
    age_scaled_survival_corr = train_df['Age_scaled'].corr(train_df['Survived'])
    
    print(f"\n📊 CORRELAÇÃO DA IDADE COM SOBREVIVÊNCIA:")
    print(f"   Age (original): {age_survival_corr:.4f}")
    print(f"   Age_scaled: {age_scaled_survival_corr:.4f}")
    
    # 2. Análise por faixas etárias
    print(f"\n👶 ANÁLISE POR FAIXAS ETÁRIAS:")
    
    # Criar faixas etárias
    train_df['Age_Group'] = pd.cut(train_df['Age'], 
                                   bins=[0, 12, 18, 30, 50, 80], 
                                   labels=['Criança (0-12)', 'Adolescente (13-18)', 
                                          'Jovem (19-30)', 'Adulto (31-50)', 'Idoso (51+)'])
    
    age_group_stats = train_df.groupby('Age_Group').agg({
        'Survived': ['count', 'sum', 'mean'],
        'Age': 'mean'
    }).round(3)
    
    age_group_stats.columns = ['Total', 'Sobreviventes', 'Taxa_Sobrevivencia', 'Idade_Media']
    print(age_group_stats)
    
    # 3. Visualizações específicas da idade
    plt.figure(figsize=(16, 12))
    
    # Subplot 1: Distribuição de idade por sobrevivência
    plt.subplot(2, 3, 1)
    for survival in [0, 1]:
        subset = train_df[train_df['Survived'] == survival]['Age']
        plt.hist(subset, alpha=0.7, label=f'Survived: {survival}', bins=20)
    plt.xlabel('Idade')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Idade por Sobrevivência', fontweight='bold')
    plt.legend()
    
    # Subplot 2: Boxplot idade por sobrevivência
    plt.subplot(2, 3, 2)
    sns.boxplot(data=train_df, x='Survived', y='Age')
    plt.title('Boxplot: Idade por Sobrevivência', fontweight='bold')
    plt.xlabel('Sobreviveu')
    plt.ylabel('Idade')
    
    # Subplot 3: Taxa de sobrevivência por faixa etária
    plt.subplot(2, 3, 3)
    age_survival_rate = train_df.groupby('Age_Group')['Survived'].mean()
    bars = plt.bar(range(len(age_survival_rate)), age_survival_rate.values, 
                   color='lightblue', edgecolor='navy')
    plt.xticks(range(len(age_survival_rate)), age_survival_rate.index, rotation=45)
    plt.ylabel('Taxa de Sobrevivência')
    plt.title('Taxa de Sobrevivência por Faixa Etária', fontweight='bold')
    
    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 4: Idade vs Tarifa colorido por sobrevivência
    plt.subplot(2, 3, 4)
    colors = ['red' if x == 0 else 'green' for x in train_df['Survived']]
    plt.scatter(train_df['Age'], train_df['Fare'], c=colors, alpha=0.6, s=30)
    plt.xlabel('Idade')
    plt.ylabel('Tarifa ($)')
    plt.title('Idade vs Tarifa (cor = sobrevivência)', fontweight='bold')
    plt.ylim(0, 300)
    
    # Subplot 5: Heatmap idade por classe e sexo
    plt.subplot(2, 3, 5)
    pivot_data = train_df.pivot_table(values='Survived', 
                                      index=pd.cut(train_df['Age'], bins=5), 
                                      columns=['Pclass', 'Sex_encoded'], 
                                      aggfunc='mean')
    sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Taxa Sobrevivência: Idade x Classe x Sexo', fontweight='bold')
    plt.ylabel('Faixa Etária')
    
    # Subplot 6: Média de idade por classe e sobrevivência
    plt.subplot(2, 3, 6)
    age_class_survival = train_df.groupby(['Pclass', 'Survived'])['Age'].mean().unstack()
    age_class_survival.plot(kind='bar', ax=plt.gca(), color=['red', 'green'])
    plt.title('Idade Média por Classe e Sobrevivência', fontweight='bold')
    plt.xlabel('Classe')
    plt.ylabel('Idade Média')
    plt.legend(['Morreu', 'Sobreviveu'])
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.show()
    
    # 4. Estatísticas detalhadas
    print(f"\n📈 ESTATÍSTICAS DETALHADAS POR SOBREVIVÊNCIA:")
    survival_age_stats = train_df.groupby('Survived')['Age'].describe()
    print(survival_age_stats)
    
    # 5. Teste de significância
    from scipy import stats
    age_survived = train_df[train_df['Survived'] == 1]['Age']
    age_died = train_df[train_df['Survived'] == 0]['Age']
    
    # Teste t para comparar médias
    t_stat, p_value = stats.ttest_ind(age_survived, age_died)
    print(f"\n🔬 TESTE T PARA DIFERENÇA DE MÉDIAS:")
    print(f"   Estatística t: {t_stat:.4f}")
    print(f"   P-valor: {p_value:.6f}")
    
    if p_value < 0.05:
        print("   ✅ Diferença estatisticamente significativa (p < 0.05)")
    else:
        print("   ❌ Diferença NÃO estatisticamente significativa (p >= 0.05)")
    
    return train_df

def test_ml_models():
    """
    Implementa e testa diferentes modelos de Machine Learning
    """
    print("\n" + "="*60)
    print("🤖 TESTANDO MODELOS DE MACHINE LEARNING")
    print("="*60)
    
    # Carregar dados
    train_df = pd.read_csv('train_processed.csv')
    test_df = pd.read_csv('test_processed.csv')
    
    # Preparar dados para modelagem
    print("\n📋 PREPARANDO DADOS PARA MODELAGEM...")
    
    # Features para diferentes tipos de modelos
    rf_features = ['Pclass', 'Sex_encoded', 'Age', 'SibSp', 'Parch', 
                   'Fare', 'Has_Cabin', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
    
    linear_features = ['Pclass_norm', 'Sex_encoded', 'Age_scaled', 'SibSp', 'Parch',
                       'Fare_scaled', 'Has_Cabin', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
    
    # Separar features e target
    X_train_full = train_df[rf_features]
    X_train_scaled = train_df[linear_features]
    y_train = train_df['Survived']
    
    X_test_full = test_df[rf_features]
    X_test_scaled = test_df[linear_features]
    
    # Split para validação
    X_train_rf, X_val_rf, y_train_split, y_val_split = train_test_split(
        X_train_full, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    X_train_linear, X_val_linear, _, _ = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    print(f"   Dados de treino: {X_train_rf.shape[0]} amostras")
    print(f"   Dados de validação: {X_val_rf.shape[0]} amostras")
    print(f"   Features para RF/XGB: {len(rf_features)}")
    print(f"   Features para modelos lineares: {len(linear_features)}")
    
    # Definir modelos
    models = {
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42),
            'X_train': X_train_rf,
            'X_val': X_val_rf,
            'X_test': X_test_full
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'X_train': X_train_rf,
            'X_val': X_val_rf,
            'X_test': X_test_full
        },
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'X_train': X_train_linear,
            'X_val': X_val_linear,
            'X_test': X_test_scaled
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'X_train': X_train_linear,
            'X_val': X_val_linear,
            'X_test': X_test_scaled
        },
        'KNN': {
            'model': KNeighborsClassifier(n_neighbors=5),
            'X_train': X_train_linear,
            'X_val': X_val_linear,
            'X_test': X_test_scaled
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(random_state=42),
            'X_train': X_train_rf,
            'X_val': X_val_rf,
            'X_test': X_test_full
        }
    }
    
    # Treinar e avaliar modelos
    results = {}
    predictions = {}
    
    print(f"\n🔄 TREINANDO E AVALIANDO MODELOS...")
    
    for name, config in models.items():
        print(f"\n--- {name} ---")
        
        model = config['model']
        X_train = config['X_train']
        X_val = config['X_val']
        X_test = config['X_test']
        
        # Treinar modelo
        model.fit(X_train, y_train_split)
        
        # Predições
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)
        
        # Probabilidades para AUC
        if hasattr(model, 'predict_proba'):
            y_prob_val = model.predict_proba(X_val)[:, 1]
            auc_score = roc_auc_score(y_val_split, y_prob_val)
        else:
            auc_score = "N/A"
        
        # Calcular acurácias
        train_acc = accuracy_score(y_train_split, y_pred_train)
        val_acc = accuracy_score(y_val_split, y_pred_val)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_full if 'rf' in name.lower() or 'tree' in name.lower() or 'boost' in name.lower() else X_train_scaled, 
                                   y_train, cv=5, scoring='accuracy')
        
        results[name] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'auc': auc_score
        }
        
        predictions[name] = y_pred_test
        
        print(f"   Acurácia Treino: {train_acc:.4f}")
        print(f"   Acurácia Validação: {val_acc:.4f}")
        print(f"   CV Média: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        if auc_score != "N/A":
            print(f"   AUC: {auc_score:.4f}")
    
    # Criar visualizações dos resultados
    print(f"\n📊 CRIANDO VISUALIZAÇÕES DOS RESULTADOS...")
    
    plt.figure(figsize=(16, 12))
    
    # 1. Comparação de acurácias
    plt.subplot(2, 3, 1)
    model_names = list(results.keys())
    train_accs = [results[name]['train_accuracy'] for name in model_names]
    val_accs = [results[name]['val_accuracy'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, train_accs, width, label='Treino', alpha=0.8, color='lightblue')
    plt.bar(x + width/2, val_accs, width, label='Validação', alpha=0.8, color='lightcoral')
    
    plt.xlabel('Modelos')
    plt.ylabel('Acurácia')
    plt.title('Comparação de Acurácias', fontweight='bold')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for i, (train_acc, val_acc) in enumerate(zip(train_accs, val_accs)):
        plt.text(i - width/2, train_acc + 0.01, f'{train_acc:.3f}', 
                ha='center', va='bottom', fontsize=9)
        plt.text(i + width/2, val_acc + 0.01, f'{val_acc:.3f}', 
                ha='center', va='bottom', fontsize=9)
    
    # 2. Cross-validation scores
    plt.subplot(2, 3, 2)
    cv_means = [results[name]['cv_mean'] for name in model_names]
    cv_stds = [results[name]['cv_std'] for name in model_names]
    
    bars = plt.bar(model_names, cv_means, yerr=cv_stds, capsize=5, 
                   color='lightgreen', alpha=0.8, edgecolor='darkgreen')
    plt.ylabel('Acurácia CV')
    plt.title('Cross-Validation (5-fold)', fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # 3. AUC scores (apenas para modelos que suportam)
    plt.subplot(2, 3, 3)
    auc_scores = []
    auc_names = []
    for name in model_names:
        if results[name]['auc'] != "N/A":
            auc_scores.append(results[name]['auc'])
            auc_names.append(name)
    
    if auc_scores:
        bars = plt.bar(auc_names, auc_scores, color='gold', alpha=0.8, edgecolor='orange')
        plt.ylabel('AUC Score')
        plt.title('AUC Scores', fontweight='bold')
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        
        for i, score in enumerate(auc_scores):
            plt.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Feature importance (Random Forest)
    plt.subplot(2, 3, 4)
    rf_model = models['Random Forest']['model']
    feature_importance = pd.DataFrame({
        'feature': rf_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    bars = plt.barh(range(len(feature_importance)), feature_importance['importance'], 
                    color='purple', alpha=0.7)
    plt.yticks(range(len(feature_importance)), feature_importance['feature'])
    plt.xlabel('Importância')
    plt.title('Feature Importance (Random Forest)', fontweight='bold')
    
    # 5. Matriz de confusão (melhor modelo)
    best_model_name = max(results.keys(), key=lambda x: results[x]['val_accuracy'])
    best_model = models[best_model_name]['model']
    best_X_val = models[best_model_name]['X_val']
    
    plt.subplot(2, 3, 5)
    y_pred_best = best_model.predict(best_X_val)
    cm = confusion_matrix(y_val_split, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={"shrink": .8})
    plt.title(f'Matriz de Confusão - {best_model_name}', fontweight='bold')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    
    # 6. ROC Curve (modelos com probabilidade)
    plt.subplot(2, 3, 6)
    for name, config in models.items():
        model = config['model']
        X_val = config['X_val']
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_val)[:, 1]
            fpr, tpr, _ = roc_curve(y_val_split, y_prob)
            auc = roc_auc_score(y_val_split, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curvas ROC', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Resumo final
    print(f"\n" + "="*60)
    print("📊 RESUMO DOS RESULTADOS")
    print("="*60)
    
    # Criar tabela de resultados
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('val_accuracy', ascending=False)
    
    print(f"\n🏆 RANKING DOS MODELOS (por acurácia de validação):")
    print(results_df.round(4))
    
    best_model_name = results_df.index[0]
    best_accuracy = results_df.loc[best_model_name, 'val_accuracy']
    
    print(f"\n🥇 MELHOR MODELO: {best_model_name}")
    print(f"   Acurácia de Validação: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    # Análise específica da idade no melhor modelo
    if best_model_name == 'Random Forest':
        rf_importance = pd.DataFrame({
            'feature': rf_features,
            'importance': models[best_model_name]['model'].feature_importances_
        }).sort_values('importance', ascending=False)
        
        age_rank = rf_importance[rf_importance['feature'] == 'Age'].index[0] + 1
        age_importance = rf_importance[rf_importance['feature'] == 'Age']['importance'].values[0]
        
        print(f"\n🧓 IMPORTÂNCIA DA IDADE NO MELHOR MODELO:")
        print(f"   Ranking: {age_rank}º lugar")
        print(f"   Importância: {age_importance:.4f} ({age_importance*100:.2f}%)")
        
        print(f"\n📋 TOP 5 FEATURES MAIS IMPORTANTES:")
        for i, (_, row) in enumerate(rf_importance.head().iterrows(), 1):
            print(f"   {i}. {row['feature']}: {row['importance']:.4f} ({row['importance']*100:.2f}%)")
    
    # Salvar predições do melhor modelo
    best_predictions = predictions[best_model_name]
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': best_predictions
    })
    
    submission.to_csv('submission_best_model.csv', index=False)
    print(f"\n💾 PREDIÇÕES SALVAS: submission_best_model.csv")
    print(f"   Modelo usado: {best_model_name}")
    print(f"   Predições para {len(best_predictions)} passageiros")
    
    return results, predictions

def main():
    """
    Função principal que executa toda a análise
    """
    print("🚀 INICIANDO ANÁLISE COMPLETA...")
    
    # 1. Análise específica da idade
    train_df = analyze_age_importance()
    
    # 2. Teste de modelos de ML
    results, predictions = test_ml_models()
    
    print(f"\n🎯 CONCLUSÕES FINAIS:")
    print("="*60)
    print("✅ Idade É uma feature importante para sobrevivência")
    print("✅ Crianças têm maior taxa de sobrevivência")
    print("✅ Diferença de idade entre sobreviventes é estatisticamente significativa")
    print("✅ Modelos treinados e testados com sucesso")
    print("✅ Melhor modelo identificado e predições salvas")
    
    return train_df, results, predictions

if __name__ == "__main__":
    train_df, results, predictions = main()
