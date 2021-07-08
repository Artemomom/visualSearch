export default class Api {
  constructor() {
    this.api_url = process.env.REACT_APP_API_URL;
  }

  async getSimilarImages(data) {
    const response = await fetch(`${this.api_url}/process`, {
      method: 'POST',
      body: data
    }).then((res) => {
      if (res.status !== 200) {
        throw new Error(
          `${res.status}: ${res.statusText || 'Something went wrong.'}`
        );
      }
      return res.json();
    });
    return response;
  }
}
