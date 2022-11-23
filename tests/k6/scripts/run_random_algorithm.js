import http from 'k6/http';
import {
    SharedArray
} from 'k6/data';
import { Rate } from "k6/metrics";
import { check } from "k6";

http.setResponseCallback(http.expectedStatuses(200, 461));      // 461 is considered a success (NotEnoughData)

const mip_engine_url = 'http://uoa.hbpmip.link:30000/algorithms';

const algorithm_names = ['anova_oneway', 'descriptive_stats', 'linear_regression', 'linear_regression_cv',
    'logistic_regression', 'logistic_regression_cv', 'pca', 'pearson_correlation', 'ttest_independent',
    'ttest_onesample', 'ttest_paired'
];

const datamodel_to_be_replaced = "dementia:0.1"
const available_datamodels = ["dementia:0.1", "dementia:0.2", "dementia:0.3"];

let algorithms_requests = [];
algorithm_names.forEach(algo_name =>
    algorithms_requests[algo_name] = new SharedArray(algo_name, function() {
        return JSON.parse(open(`/algorithm_requests/${algo_name}_expected.json`))["test_cases"];
    })
);

export let errorRate = new Rate("errors");

export default function() {
    const algorithm_names = Object.keys(algorithms_requests);
    const algorithm_name = algorithm_names[Math.floor(Math.random() * algorithm_names.length)];

    const algorithm_requests = algorithms_requests[algorithm_name];
    const algorithm_request = algorithm_requests[Math.floor(Math.random() * algorithm_requests.length)];
    const algorithm_input = JSON.stringify(algorithm_request["input"]);

    const datamodel = available_datamodels[Math.floor(Math.random() * available_datamodels.length)];
    const algorithm_input_with_random_datamodel = algorithm_input.replace(datamodel_to_be_replaced,datamodel);

    const algorithm_url = mip_engine_url + `/${algorithm_name}`;
    const request_params = {
        headers: {
            'Content-type': 'application/json',
            'Accept': 'text/plain'
        },
        timeout: '3600s'
    };

    const res = http.post(algorithm_url, algorithm_input_with_random_datamodel, request_params);
    console.log(`${algorithm_url}`);

    let success = check(res, {"is status 200 or 461": (r) => r.status === 200 || r.status === 461});
    if (!success) {
        errorRate.add(res.status)
        console.log(`Status: ${res.status} \nResponse: ${res.body}`)
    }
}
