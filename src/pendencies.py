import pandas as pd

def identificar_pendencias(pendencias_df):
    """Processa pendências já existentes no sistema."""
    # Exemplo: classificar por tipo e prioridade
    pendencias_df['prioridade'] = pendencias_df['tipo'].apply(lambda x: 'Alta' if x in ['Estorno'] else 'Média')
    return pendencias_df.sort_values('prioridade', ascending=False)

def gerar_planilha_ajustes(pendencias_df, output_path='reports/pendencias_para_ajuste.xlsx'):
    """Gera uma planilha com as pendências a serem tratadas."""
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        pendencias_df.to_excel(writer, sheet_name='Pendencias', index=False)
        # Formatação básica
        workbook = writer.book
        worksheet = writer.sheets['Pendencias']
        # Ajustar largura das colunas
        for i, col in enumerate(pendencias_df.columns):
            max_len = max(pendencias_df[col].astype(str).map(len).max(), len(col)) + 2
            worksheet.set_column(i, i, max_len)
    print(f"Planilha gerada em {output_path}")