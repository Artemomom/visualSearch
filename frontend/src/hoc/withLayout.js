import React from 'react';
import TopNavigation from '../components/TopNavigation/TopNavigation';

export function withLayout(WrappedComponent) {
  return function Component(props) {
    return (
      <div
        style={{
          height: '100vh'
        }}>
        <TopNavigation {...props} />
        <div
          style={{
            margin: '0 10%',
            width: 'auto',
            height: '100%',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-around'
          }}>
          <WrappedComponent {...props} />
        </div>
      </div>
    );
  };
}
