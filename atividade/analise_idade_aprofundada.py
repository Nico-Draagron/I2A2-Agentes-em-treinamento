import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def deeper_age_analysis():
    """
    Análise mais profunda da relação idade-sobrevivência
    """
    print("="*70)
    print("🔍 ANÁLISE APROFUNDADA: POR QUE A CORRELAÇÃO IDADE É BAIXA?")
    print("="*70)
    
    # Carregar dados
    train_df = pd.read_csv('train_processed.csv')
    
    print(f"\n🧐 INVESTIGANDO A APARENTE BAIXA CORRELAÇÃO...")
    
    # 1. Análise por grupos
    print(f"\n1️⃣ EFEITO DA IDADE POR SUBGRUPOS:")
    
    # Por sexo
    print(f"\n👫 POR SEXO:")
    for sex in [0, 1]:  # 0=female, 1=male
        sex_name = "Feminino" if sex == 0 else "Masculino"
        subset = train_df[train_df['Sex_encoded'] == sex]
        age_corr = subset['Age'].corr(subset['Survived'])
        print(f"   {sex_name}: correlação = {age_corr:.4f}")
        
        # Estatísticas de idade por sobrevivência
        for survival in [0, 1]:
            survival_name = "Morreu" if survival == 0 else "Sobreviveu"
            age_stats = subset[subset['Survived'] == survival]['Age']
            print(f"     {survival_name}: média = {age_stats.mean():.1f}, mediana = {age_stats.median():.1f}")
    
    # Por classe
    print(f"\n🎭 POR CLASSE:")
    for pclass in [1, 2, 3]:
        subset = train_df[train_df['Pclass'] == pclass]
        age_corr = subset['Age'].corr(subset['Survived'])
        print(f"   Classe {pclass}: correlação = {age_corr:.4f}")
        
        # Estatísticas de idade por sobrevivência
        for survival in [0, 1]:
            survival_name = "Morreu" if survival == 0 else "Sobreviveu"
            age_stats = subset[subset['Survived'] == survival]['Age']
            if len(age_stats) > 0:
                print(f"     {survival_name}: média = {age_stats.mean():.1f}, mediana = {age_stats.median():.1f}")
    
    # 2. Análise de faixas etárias específicas
    print(f"\n2️⃣ PADRÕES POR FAIXAS ETÁRIAS ESPECÍFICAS:")
    
    # Definir faixas mais granulares
    bins = [0, 5, 12, 18, 25, 35, 50, 65, 100]
    labels = ['Bebê (0-5)', 'Criança (6-12)', 'Adolescente (13-18)', 
              'Jovem (19-25)', 'Adulto Jovem (26-35)', 'Adulto (36-50)', 
              'Meia-idade (51-65)', 'Idoso (65+)']
    
    train_df['Age_Detailed'] = pd.cut(train_df['Age'], bins=bins, labels=labels, include_lowest=True)
    
    age_detailed_stats = train_df.groupby('Age_Detailed').agg({
        'Survived': ['count', 'sum', 'mean'],
        'Age': 'mean'
    }).round(3)
    
    age_detailed_stats.columns = ['Total', 'Sobreviventes', 'Taxa_Sobrevivencia', 'Idade_Media']
    print(age_detailed_stats)
    
    # 3. Análise de outliers e padrões especiais
    print(f"\n3️⃣ PADRÕES ESPECIAIS:")
    
    # Crianças vs resto
    children = train_df[train_df['Age'] <= 12]
    adults = train_df[train_df['Age'] > 12]
    
    children_survival = children['Survived'].mean()
    adults_survival = adults['Survived'].mean()
    
    print(f"\n👶 CRIANÇAS (≤12 anos) vs DEMAIS:")
    print(f"   Crianças: {len(children)} pessoas, sobrevivência = {children_survival:.1%}")
    print(f"   Demais: {len(adults)} pessoas, sobrevivência = {adults_survival:.1%}")
    print(f"   Diferença: {children_survival - adults_survival:.1%}")
    
    # Teste estatístico
    chi2, p_value = stats.chi2_contingency(pd.crosstab(train_df['Age'] <= 12, train_df['Survived']))[:2]
    print(f"   Teste Chi-quadrado: p-valor = {p_value:.6f}")
    if p_value < 0.05:
        print("   ✅ Diferença estatisticamente significativa!")
    
    # Idosos
    elderly = train_df[train_df['Age'] >= 60]
    if len(elderly) > 0:
        elderly_survival = elderly['Survived'].mean()
        print(f"\n👴 IDOSOS (≥60 anos):")
        print(f"   {len(elderly)} pessoas, sobrevivência = {elderly_survival:.1%}")
    
    # 4. Visualizações detalhadas
    plt.figure(figsize=(20, 15))
    
    # Subplot 1: Taxa de sobrevivência por idade (suavizada)
    plt.subplot(3, 4, 1)
    
    # Calcular taxa de sobrevivência por idade usando rolling window
    age_survival_smooth = []
    ages = sorted(train_df['Age'].unique())
    
    for age in ages:
        # Janela de ±2 anos
        window_data = train_df[(train_df['Age'] >= age-2) & (train_df['Age'] <= age+2)]
        if len(window_data) >= 5:  # Mínimo de 5 pessoas
            survival_rate = window_data['Survived'].mean()
            age_survival_smooth.append((age, survival_rate))
    
    if age_survival_smooth:
        ages_smooth, rates_smooth = zip(*age_survival_smooth)
        plt.plot(ages_smooth, rates_smooth, 'b-', linewidth=2, alpha=0.8)
        plt.scatter(ages_smooth, rates_smooth, alpha=0.6, s=30)
    
    plt.xlabel('Idade')
    plt.ylabel('Taxa de Sobrevivência')
    plt.title('Taxa de Sobrevivência Suavizada por Idade', fontweight='bold')
    plt.grid(alpha=0.3)
    
    # Subplot 2: Boxplot idade por sobrevivência e sexo
    plt.subplot(3, 4, 2)
    data_plot = train_df.copy()
    data_plot['Sex_Survival'] = data_plot['Sex_encoded'].astype(str) + '_' + data_plot['Survived'].astype(str)
    sex_survival_labels = {'0_0': 'F_Morreu', '0_1': 'F_Sobreviveu', '1_0': 'M_Morreu', '1_1': 'M_Sobreviveu'}
    data_plot['Sex_Survival'] = data_plot['Sex_Survival'].map(sex_survival_labels)
    
    sns.boxplot(data=data_plot, x='Sex_Survival', y='Age', palette='Set2')
    plt.title('Idade por Sexo e Sobrevivência', fontweight='bold')
    plt.xticks(rotation=45)
    
    # Subplot 3: Heatmap idade x classe x sobrevivência
    plt.subplot(3, 4, 3)
    
    # Criar bins de idade para heatmap
    train_df['Age_Bins'] = pd.cut(train_df['Age'], bins=6, labels=['0-13', '14-26', '27-39', '40-52', '53-65', '66+'])
    
    heatmap_data = train_df.pivot_table(values='Survived', 
                                        index='Age_Bins', 
                                        columns='Pclass', 
                                        aggfunc='mean')
    
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Taxa Sobrevivência: Idade x Classe', fontweight='bold')
    plt.ylabel('Faixa Etária')
    
    # Subplot 4: Distribuição de idade colorida por sobrevivência
    plt.subplot(3, 4, 4)
    
    # Histograma empilhado
    bins = np.arange(0, 81, 5)
    ages_died = train_df[train_df['Survived'] == 0]['Age']
    ages_survived = train_df[train_df['Survived'] == 1]['Age']
    
    plt.hist([ages_died, ages_survived], bins=bins, label=['Morreu', 'Sobreviveu'], 
             alpha=0.7, color=['red', 'green'], stacked=True)
    plt.xlabel('Idade')
    plt.ylabel('Frequência')
    plt.title('Distribuição de Idade Empilhada', fontweight='bold')
    plt.legend()
    
    # Subplot 5: Taxa de sobrevivência por faixa etária detalhada
    plt.subplot(3, 4, 5)
    detailed_survival = train_df.groupby('Age_Detailed')['Survived'].mean()
    
    bars = plt.bar(range(len(detailed_survival)), detailed_survival.values, 
                   color='lightblue', edgecolor='navy', alpha=0.8)
    plt.xticks(range(len(detailed_survival)), detailed_survival.index, rotation=45)
    plt.ylabel('Taxa de Sobrevivência')
    plt.title('Taxa por Faixa Etária Detalhada', fontweight='bold')
    
    # Adicionar valores nas barras
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    # Subplot 6: Scatter plot idade vs tarifa colorido por sobrevivência
    plt.subplot(3, 4, 6)
    colors = ['red' if x == 0 else 'green' for x in train_df['Survived']]
    sizes = [20 if x <= 12 else 10 for x in train_df['Age']]  # Destacar crianças
    
    plt.scatter(train_df['Age'], train_df['Fare'], c=colors, s=sizes, alpha=0.6)
    plt.xlabel('Idade')
    plt.ylabel('Tarifa ($)')
    plt.title('Idade vs Tarifa (crianças destacadas)', fontweight='bold')
    plt.ylim(0, 300)
    
    # Subplot 7: Análise de família - idade vs sobrevivência
    plt.subplot(3, 4, 7)
    
    # Criar indicador de família (SibSp + Parch > 0)
    train_df['Has_Family'] = (train_df['SibSp'] + train_df['Parch'] > 0).astype(int)
    
    family_age_survival = train_df.groupby(['Has_Family', 'Age_Bins'])['Survived'].mean().unstack()
    
    family_age_survival.plot(kind='bar', ax=plt.gca(), color=['lightcoral', 'lightblue'])
    plt.title('Sobrevivência: Família x Idade', fontweight='bold')
    plt.xlabel('Tem Família (0=Não, 1=Sim)')
    plt.ylabel('Taxa de Sobrevivência')
    plt.legend(title='Faixa Etária', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    
    # Subplot 8: Curva de sobrevivência por idade (estilo survival analysis)
    plt.subplot(3, 4, 8)
    
    # Calcular taxa cumulativa de sobrevivência
    age_counts = train_df.groupby('Age').agg({'Survived': ['count', 'sum']}).reset_index()
    age_counts.columns = ['Age', 'Total', 'Survived']
    age_counts['Cumulative_Total'] = age_counts['Total'].cumsum()
    age_counts['Cumulative_Survived'] = age_counts['Survived'].cumsum()
    age_counts['Survival_Rate'] = age_counts['Cumulative_Survived'] / age_counts['Cumulative_Total']
    
    plt.plot(age_counts['Age'], age_counts['Survival_Rate'], 'b-', linewidth=2)
    plt.xlabel('Idade')
    plt.ylabel('Taxa de Sobrevivência Cumulativa')
    plt.title('Curva de Sobrevivência Cumulativa', fontweight='bold')
    plt.grid(alpha=0.3)
    
    # Subplot 9: Comparação crianças vs adultos por classe
    plt.subplot(3, 4, 9)
    
    comparison_data = []
    for pclass in [1, 2, 3]:
        class_data = train_df[train_df['Pclass'] == pclass]
        children_class = class_data[class_data['Age'] <= 12]['Survived'].mean()
        adults_class = class_data[class_data['Age'] > 12]['Survived'].mean()
        
        comparison_data.extend([
            ('Classe ' + str(pclass), 'Crianças', children_class),
            ('Classe ' + str(pclass), 'Adultos', adults_class)
        ])
    
    comp_df = pd.DataFrame(comparison_data, columns=['Classe', 'Grupo', 'Taxa_Sobrevivencia'])
    
    pivot_comp = comp_df.pivot(index='Classe', columns='Grupo', values='Taxa_Sobrevivencia')
    pivot_comp.plot(kind='bar', ax=plt.gca(), color=['lightblue', 'lightcoral'])
    plt.title('Crianças vs Adultos por Classe', fontweight='bold')
    plt.ylabel('Taxa de Sobrevivência')
    plt.xticks(rotation=0)
    plt.legend()
    
    # Subplot 10: Densidade de idade por sobrevivência
    plt.subplot(3, 4, 10)
    
    for survival in [0, 1]:
        subset = train_df[train_df['Survived'] == survival]['Age']
        sns.kdeplot(subset, label=f'Survived: {survival}', alpha=0.7)
    
    plt.xlabel('Idade')
    plt.ylabel('Densidade')
    plt.title('Densidade de Idade por Sobrevivência', fontweight='bold')
    plt.legend()
    
    # Subplot 11: Idade média por combinações de fatores
    plt.subplot(3, 4, 11)
    
    factor_combinations = train_df.groupby(['Sex_encoded', 'Pclass', 'Survived'])['Age'].mean().unstack()
    
    sns.heatmap(factor_combinations, annot=True, cmap='viridis', fmt='.1f', cbar_kws={"shrink": .8})
    plt.title('Idade Média: Sexo x Classe x Sobrevivência', fontweight='bold')
    plt.ylabel('(Sexo, Classe)')
    
    # Subplot 12: Estatísticas resumo
    plt.subplot(3, 4, 12)
    
    # Texto com estatísticas importantes
    stats_text = f"""
ESTATÍSTICAS CHAVE:

Correlação Idade-Sobrevivência: {train_df['Age'].corr(train_df['Survived']):.4f}

Por Faixa Etária:
• Crianças (≤12): {children_survival:.1%}
• Adultos (>12): {adults_survival:.1%}
• Diferença: {children_survival - adults_survival:.1%}

Por Sexo:
• Feminino: {train_df[train_df['Sex_encoded']==0]['Age'].corr(train_df[train_df['Sex_encoded']==0]['Survived']):.4f}
• Masculino: {train_df[train_df['Sex_encoded']==1]['Age'].corr(train_df[train_df['Sex_encoded']==1]['Survived']):.4f}

Idade Média:
• Sobreviventes: {train_df[train_df['Survived']==1]['Age'].mean():.1f}
• Não sobreviventes: {train_df[train_df['Survived']==0]['Age'].mean():.1f}

P-valor (crianças vs adultos): {p_value:.6f}
    """
    
    plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    plt.axis('off')
    plt.title('Resumo Estatístico', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # 5. Conclusões
    print(f"\n" + "="*70)
    print("🎯 CONCLUSÕES SOBRE A IMPORTÂNCIA DA IDADE")
    print("="*70)
    
    print(f"\n✅ POR QUE A CORRELAÇÃO PARECE BAIXA?")
    print(f"   1. Efeito não-linear: crianças têm alta sobrevivência, mas a relação não é linear")
    print(f"   2. Interação com outras variáveis: sexo e classe modificam o efeito da idade")
    print(f"   3. Distribuição desigual: poucos idosos e crianças comparado a adultos")
    
    print(f"\n✅ EVIDÊNCIAS DA IMPORTÂNCIA DA IDADE:")
    print(f"   • Crianças (≤12) têm {children_survival:.1%} de sobrevivência vs {adults_survival:.1%} dos demais")
    print(f"   • Diferença de {children_survival - adults_survival:.1%} é substancial")
    print(f"   • Padrão 'Women and children first' claramente visível")
    print(f"   • Idade interage com classe social e sexo")
    
    print(f"\n✅ RECOMENDAÇÕES PARA MODELAGEM:")
    print(f"   • Criar variável categórica para crianças (≤12 anos)")
    print(f"   • Considerar interações idade*sexo e idade*classe")
    print(f"   • Usar transformações não-lineares da idade")
    print(f"   • Age é definitivamente uma feature importante!")
    
    return train_df

if __name__ == "__main__":
    train_df = deeper_age_analysis()
