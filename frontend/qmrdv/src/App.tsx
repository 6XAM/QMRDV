import React from 'react';
import './App.css';

const WebSocketURI: string = 'wss://vrrvgmqs2g.execute-api.us-east-1.amazonaws.com/Test'

class App extends React.Component {
  state: {
    ws: WebSocket | null;
    ready: boolean;
  }
  constructor(props: any) {
    super(props);
    this.state = {
      ws: null,
      ready: false
    }
  }
  connect(): void {
    let ws = new WebSocket(WebSocketURI)
    if (this.state.ws !== null) {
      this.state.ws.close();
    }
    this.setState({
      ws: ws,
      ready: false
    });
    ws.onmessage = (ev: MessageEvent): any => {
      console.log(JSON.parse(ev.data));
    };
    ws.onopen = (ev: Event): any => {
      this.setState({
        ready: true
      });
    };
  }
  compute(): void {
    if (this.state.ws !== null && this.state.ws.readyState === WebSocket.OPEN) {
      this.state.ws.send(JSON.stringify({
        action: 'compute',
        test: 123123
      }));
    }
  }
  disconnect(): void {
    if (this.state.ws !== null) {
      this.state.ws.close();
    }
    this.setState({
      ws: null,
      ready: false
    });
  }
  render(): React.ReactElement {
    return (
      <div className="App">
        <button onClick={this.connect.bind(this)}>
          {this.state.ws ? 'Reconnect' : 'Connect'}
        </button>
        <button disabled={(!this.state.ws || !this.state.ready)} onClick={this.compute.bind(this)}>
          Compute
        </button>
        <button disabled={!this.state.ws} onClick={this.disconnect.bind(this)}>
          Disconnect
        </button>
      </div>
    );
  }
}

export default App;
