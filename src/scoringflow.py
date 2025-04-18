from metaflow import FlowSpec, step, Parameter

class DiabetesScoringFlow(FlowSpec):

    random_state = Parameter("random_state", default=42)

    @step
    def start(self):
        import dataprocessing

        # Load test data
        self.X_test, self.y_test = dataprocessing.get_test_data(
            random_state=self.random_state
        )
        print('Test data shape:', self.X_test.shape)
        self.next(self.load_model)

    @step
    def load_model(self):
        import mlflow
        import mlflow.sklearn

        # Load best diabetes model from MLFlow
        mlflow.set_tracking_uri('https://mlops-4-960398271618.us-west2.run.app')
        print('Loading model from MLFlow ...')
        self.model = mlflow.sklearn.load_model('models:/metaflow-diabetes-model/5')

        self.next(self.end)

    @step
    def end(self):
        from sklearn.metrics import mean_squared_error

        y_pred = self.model.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)

        # Print the predicted values
        print('Model', self.model)
        print('Sample predictions:', y_pred[:5])
        print('Test set MSE:', round(mse, 4))

if __name__ == '__main__':
    DiabetesScoringFlow()