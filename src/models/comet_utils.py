# MIT License
# Copyright 2020 Ryan Hausen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# ofthis software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from itertools import starmap
from typing import Callable, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

import comet_ml


MetricFuncResult = Dict[str, Tuple[str, Union[float, plt.Figure, np.ndarray]]]


def setup_experiment(
    key: str, project_name: str, params: Dict[str, str], code_file: str, disabled: bool
) -> comet_ml.Experiment:

    if key:
        experiment = comet_ml.ExistingExperiment(
            api_key=os.getenv("comet_key"), comet_experiment_key=key,
        )
    else:
        experiment = comet_ml.Experiment(
            api_key=os.getenv("comet_key"),
            project_name=project_name,
            disabled=disabled,
            auto_metric_logging=False,
        )
        experiment.log_parameters(params)
        experiment.set_code(filename=code_file, overwrite=True)

    return experiment


def get_async_metric_logging_f(
    experiment: comet_ml.Experiment,
    experiment_context: Callable,
    step:int,
    ) -> Callable[MetricFuncResult, None]:

    def log_metric(
        metric:str,
        value:Tuple[str, Union[float, plt.Figure, np.ndarray]]
    ) -> None:
        """Function that logs a single metric."""
        metric_type, metric_value = value

        if metric_type=="float":
            experiment.log_metric(metric, metric_value, step=step)
        elif metric_type=="figure":
            experiment.log_figure(
                figure_name=metric,
                figure=metric_value,
                step=step
            )
        elif metric_type=="image":
            experiment.log_image(metric_value, name=metric, step=step)
        else:
            msg = "Invalid Metric Type. Must be ['float', 'figure', or 'image']"
            raise ValueError(msg)

        return None

    def record_metric_func_result(result:MetricFuncResult) -> None:
        """Records the metrics in the given context"""
        with experiment_context():
            for _ in starmap(log_metric, map(lambda k: (k, result[k]), result)):
                pass

    return record_metric_func_result
