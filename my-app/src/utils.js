import querystring from 'querystring';
// import config from "./config.json"

const API_ROOT = 'https://j6ykpxe4hl.execute-api.eu-west-2.amazonaws.com/prod/';

export const GET = async (resource, additional_params) => {
    const options = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    const url = additional_params
        ? `${API_ROOT}${resource}?${querystring.encode(additional_params)}`
        : `${API_ROOT}${resource}`;
    //console.debug('STREAMLINE GETting', url);
    let response = await fetch(url, options);
    response = await response.json();
    //console.debug('response from', resource, ':', response);
    return response;
};

export const POST = async (resource, body) => {
    const options = {
        method: 'POST',
        mode: 'cors',
        body: JSON.stringify(body),
        headers: {
            'Content-Type': 'application/json',
        },
    };
    const url = API_ROOT + resource;
    let response = await fetch(url, options);
    response = await response.json();
    //console.debug('response from', resource, ':', response);
    return response;
};

export const makeid = (length = 8) => {
    let text = '';
    const possible = 'abcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < length; i++) {
        text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
};