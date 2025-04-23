from metaflow import FlowSpec, step, Parameter, conda_base, resources, retry, catch, timeout

@conda_base(libraries={'numpy':'1.23.5', 'scikit-learn':'1.2.2', 'mlflow':'2.9.2', 'databricks-cli': '0.17.6'}, python='3.9.16')
class DiabetesTrainingFlow(FlowSpec):

    alpha = Parameter("alpha", default=1.0)
    random_state = Parameter("random_state", default=42)

    @resources(cpu=2, memory=2048)
    @timeout(seconds=300)
    @retry(times=2)
    @catch(var="error_start")
    @step
    def start(self):
        import dataprocessing

        self.X_train, self.X_val, self.y_train, self.y_val = dataprocessing.load_data(
            random_state=self.random_state
        )
        self.next(self.train_ridge, self.train_lasso)

    @resources(cpu=2, memory=2048)
    @timeout(seconds=180)
    @retry(times=2)
    @catch(var="error_ridge")
    @step
    def train_ridge(self):
        from sklearn.linear_model import Ridge

        self.model_name = "Ridge"
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(self.X_train, self.y_train)
        self.next(self.choose_model)

    @resources(cpu=2, memory=2048)
    @timeout(seconds=180)
    @retry(times=2)
    @catch(var="error_lasso")
    @step
    def train_lasso(self):
        from sklearn.linear_model import Lasso

        self.model_name = "Lasso"
        self.model = Lasso(alpha=self.alpha)
        self.model.fit(self.X_train, self.y_train)
        self.next(self.choose_model)

    @resources(cpu=2, memory=2048)
    @timeout(seconds=300)
    @retry(times=2)
    @catch(var="error_choose_model")
    @step
    def choose_model(self, inputs):
        import mlflow
        import mlflow.sklearn
        from sklearn.metrics import mean_squared_error

        mlflow.set_tracking_uri('https://mlflow-v1-928344033488.us-west2.run.app')
        mlflow.set_experiment('metaflow-project')

        def score(inp):
            y_pred = inp.model.predict(inp.X_val)
            mse = mean_squared_error(inp.y_val, y_pred)
            return inp.model_name, inp.model, mse

        self.results = sorted(map(score, inputs), key=lambda x: x[2])
        self.model_name, self.model, self.mse = self.results[0]

        with mlflow.start_run():
            mlflow.sklearn.log_model(
                self.model,
                artifact_path='metaflow_train',
                registered_model_name="metaflow-diabetes-model"
            )
            mlflow.end_run()

        self.next(self.end)

    @resources(cpu=2, memory=2048)
    @step
    def end(self):
        print('Scores:')
        print('\n'.join('%s %.4f' % (name, mse) for name, model, mse in self.results))
        print('Model:', self.model)

if __name__ == '__main__':
    DiabetesTrainingFlow()