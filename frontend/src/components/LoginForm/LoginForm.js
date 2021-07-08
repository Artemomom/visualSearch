import React, { useState } from 'react';
import PropTypes from 'prop-types';
import { useHistory } from 'react-router-dom';
import TextField from '@material-ui/core/TextField';
import Button from '../Button';

import './LoginForm.scss';
import { validateEmail } from '../../utils/validations';

const LoginForm = ({ setIsAuth }) => {
  const [email, setEmail] = useState({ value: '', error: false });
  const [password, setPassword] = useState({ value: '', error: false });
  const history = useHistory();

  const login = () => {
    setIsAuth(true);
    history.push('/');
  };

  const onEmailChange = (e) => {
    const value = e.target.value;
    const isValid = validateEmail(value);
    setEmail({ value, error: !isValid });
  };

  const onPasswordChange = (e) => {
    const value = e.target.value;
    const isValid = value.length >= 8 && value.length <= 30;
    setPassword({ value, error: !isValid });
  };

  const isButtonDisabled = (email, password) => {
    return email.error || password.error || email.value === '' || password.value === '';
  };

  return (
    <div className="login-form">
      <span className="title">Welcome</span>
      <p className="message">Sign in to continue</p>
      <form noValidate autoComplete="off" style={{ display: 'flex', flexDirection: 'column' }}>
        <TextField
          id="standard-basic"
          error={email.error}
          label="Email"
          margin="dense"
          value={email.value}
          onChange={onEmailChange}
          helperText={email.error ? 'Email is not valid' : undefined}
        />
        <TextField
          id="standard-basic"
          error={password.error}
          label="Password"
          type="password"
          margin="dense"
          value={password.value}
          onChange={onPasswordChange}
          helperText={
            password.error ? 'Your password must be between 8 and 30 characters' : undefined
          }
        />
        <div style={{ marginTop: 20 }}>
          <Button
            title="Login"
            onClick={login}
            disabled={isButtonDisabled(email, password)}
            padding="10px 30px"
          />
        </div>
      </form>
    </div>
  );
};

LoginForm.propTypes = {
  setIsAuth: PropTypes.func
};

export default LoginForm;
