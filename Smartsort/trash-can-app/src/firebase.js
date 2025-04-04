// import firebase from 'firebase';
// //Swap this out for your own firebase config
// const config = {
//   apiKey: "____________________________________",
//   authDomain: "smarttrash11810.firebaseapp.com",
//   databaseURL: 'https://smarttrash-11810-default-rtdb.firebaseio.com/',
//   projectId: "smarttrash11810",
//   storageBucket: "smarttrash11810.appspot.com",
//   messagingSenderId: "610217870116",
//   appId: "1:610217870116:web:f411ee6e86a970977f69b1",
//   measurementId: "G-1SZ2PNY4RK"
// };
// firebase.initializeApp(config);
// export const provider = new firebase.auth.GoogleAuthProvider();
// export const auth = firebase.auth();
// export default firebase;

// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getAnalytics } from "firebase/analytics";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyC0FQBLiKXbX6vtW2JOidckyXBDTmEf168",
  authDomain: "smarttrash-11810.firebaseapp.com",
  databaseURL: "https://smarttrash-11810-default-rtdb.firebaseio.com",
  projectId: "smarttrash-11810",
  storageBucket: "smarttrash-11810.firebasestorage.app",
  messagingSenderId: "93495635791",
  appId: "1:93495635791:web:8aa232583aceb0724adf3a",
  measurementId: "G-RCZFXPCE25"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const analytics = getAnalytics(app);
