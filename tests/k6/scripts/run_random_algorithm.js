import http from 'k6/http';
import {
    SharedArray
} from 'k6/data';

http.setResponseCallback(http.expectedStatuses(200, 461));      // 461 is considered a success (NotEnoughData)

const mip_engine_url = 'http://uoa.hbpmip.link:30000/algorithms';

const algorithm_names = ['anova_oneway', 'descriptive_stats', 'linear_regression', 'linear_regression_cv',
    'logistic_regression', 'logistic_regression_cv', 'pca', 'pearson_correlation', 'ttest_independent',
    'ttest_onesample', 'ttest_paired'
];

let algorithms_requests = [];
algorithm_names.forEach(algo_name =>
    algorithms_requests[algo_name] = new SharedArray(algo_name, function() {
        return JSON.parse(open(`/algorithm_requests/${algo_name}_expected.json`))["test_cases"];
    })
);


export default function() {
    const algorithm_names = Object.keys(algorithms_requests);
    const algorithm_name = algorithm_names[Math.floor(Math.random() * algorithm_names.length)];

    const algorithm_requests = algorithms_requests[algorithm_name];
    const algorithm_request = algorithm_requests[Math.floor(Math.random() * algorithm_requests.length)];

    const algorithm_url = mip_engine_url + `/${algorithm_name}`;
    const request_params = {
        headers: {
            'Content-type': 'application/json',
            'Accept': 'text/plain'
        }
    };

    http.post(algorithm_url, JSON.stringify(algorithm_request["input"]), request_params);
    console.log(`${algorithm_url}`);
}
