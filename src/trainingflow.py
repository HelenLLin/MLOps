from metaflow import FlowSpec, step, Parameter

class DiabetesTrainingFlow(FlowSpec):

    alpha = Parameter("alpha", default=1.0)
    random_state = Parameter("random_state", default=42)

    @step
    def start(self):
        import dataprocessing

        # Preprocess and split data
        self.X_train, self.X_val, self.y_train, self.y_val = dataprocessing.load_data(
            random_state=self.random_state
        )
        self.next(self.train_ridge, self.train_lasso)

    @step
    def train_ridge(self):
        from sklearn.linear_model import Ridge

        # Fit Ridge linear model
        self.model_name = "Ridge"
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(self.X_train, self.y_train)
        self.next(self.choose_model)

    @step
    def train_lasso(self):
        from sklearn.linear_model import Lasso

        # Fit Lasso linear model
        self.model_name = "Lasso"
        self.model = Lasso(alpha=self.alpha)
        self.model.fit(self.X_train, self.y_train)
        self.next(self.choose_model)

    @step
    def choose_model(self, inputs):
        import mlflow
        import mlflow.sklearn
        from sklearn.metrics import mean_squared_error

        mlflow.set_tracking_uri('https://mlops-4-960398271618.us-west2.run.app')
        mlflow.set_experiment('metaflow-project')

        # Function to compare the predicted and true for each model
        def score(inp):
            y_pred = inp.model.predict(inp.X_val)
            mse = mean_squared_error(inp.y_val, y_pred)
            return inp.model_name, inp.model, mse
        
        # Sort the results by MSE
        self.results = sorted(map(score, inputs), key=lambda x: x[2])
        self.model_name, self.model, self.mse = self.results[0]
        with mlflow.start_run():
            # Log the model with the best MSE
            mlflow.sklearn.log_model(self.model, artifact_path = 'metaflow_train', registered_model_name="metaflow-diabetes-model")
            mlflow.end_run()
        self.next(self.end)

    @step
    def end(self):
        # Print the model MSEs
        print('Scores:')
        print('\n'.join('%s %.4f' % (name, rmse) for name, model, rmse in self.results))
        print('Model:', self.model)

if __name__ == '__main__':
    DiabetesTrainingFlow()