import sys
sys.path.insert(0, ".")

import pandas as pd
from sklearn.model_selection import train_test_split

SEED      = 42
TEST_SIZE = 0.2

if __name__ == '__main__':
    df = pd.read_excel('processed/train_extracted_features.xlsx')

    # Aleatoriza e divide mantendo a proporção por objeto
    df_train, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df['object_id'],
    )

    df_train = df_train.copy()
    df_test  = df_test.copy()

    df_train['set'] = 'train'
    df_test['set']  = 'test'

    df_out = pd.concat([df_train, df_test], ignore_index=True)
    df_out = df_out.sort_values(['object_id', 'frame_id']).reset_index(drop=True)

    df_out.to_excel('processed/splitted_train.xlsx', index=False)

    print(f'Total de amostras: {len(df_out)}')
    print(f'Treino:            {len(df_train)} ({len(df_train)/len(df_out)*100:.1f}%)')
    print(f'Teste:             {len(df_test)} ({len(df_test)/len(df_out)*100:.1f}%)')
    print('\nDistribuição por objeto:')
    print(df_out.groupby(['object_id', 'set']).size().unstack(fill_value=0))