x = pd.read_csv('/content/c_50.csv', header=None).sample(100)
y = pd.read_csv('/content/r1_50.csv', header=None).sample(100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = args.train_test_split, shuffle=False) 


dataset = FinancialDataset(data_dir, input_dim=args.input_dim, linear=args.linear)