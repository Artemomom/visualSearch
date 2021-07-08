import React from 'react';
import PropTypes from 'prop-types';
import { withLayout } from '../hoc/withLayout';
import LoginForm from '../components/LoginForm/LoginForm';

const Login = ({ setIsAuth }) => {
  return <LoginForm setIsAuth={setIsAuth} />;
};

Login.propTypes = {
  setIsAuth: PropTypes.func
};

export default withLayout(Login);
